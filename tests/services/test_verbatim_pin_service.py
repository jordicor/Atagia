"""Tests for verbatim pin service behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.space_repository import SpaceRepository
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    SpaceBoundaryMode,
    VerbatimPinStatus,
    VerbatimPinTargetKind,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.verbatim_pin_service import VerbatimPinService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class NoopProvider(LLMProvider):
    name = "noop-verbatim-pin-tests"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError(f"LLM should not be called in verbatim pin tests: {request.metadata}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings should not be called in verbatim pin tests: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-verbatim-pin.db"),
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
    runtime.clock = FrozenClock(datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc))
    return runtime


@pytest.mark.asyncio
async def test_verbatim_pin_service_resolves_message_memory_object_and_text_span_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        connection = await runtime.open_connection()
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            service = VerbatimPinService(runtime)

            await users.create_user("usr_1")
            await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")
            await messages.create_message(
                "msg_1",
                "cnv_1",
                "user",
                1,
                "alpha beta gamma",
                3,
                {},
                "2026-03-30T11:00:00+00:00",
            )
            await memories.create_memory_object(
                user_id="usr_1",
                workspace_id=None,
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="memory object snapshot",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.8,
                privacy_level=0,
                intimacy_boundary=IntimacyBoundary.ROMANTIC_PRIVATE,
                intimacy_boundary_confidence=0.91,
                memory_id="mem_1",
            )

            message_pin = await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.MESSAGE,
                target_id="msg_1",
                privacy_level=0,
                created_by="usr_1",
            )
            memory_pin = await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.MEMORY_OBJECT,
                target_id="mem_1",
                privacy_level=0,
                created_by="usr_1",
            )
            span_pin = await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.TEXT_SPAN,
                target_id="msg_1",
                target_span_start=6,
                target_span_end=10,
                privacy_level=0,
                created_by="usr_1",
            )

            assert message_pin["canonical_text"] == "alpha beta gamma"
            assert memory_pin["canonical_text"] == "memory object snapshot"
            assert memory_pin["intimacy_boundary"] == "romantic_private"
            assert memory_pin["intimacy_boundary_confidence"] == 0.91
            assert memory_pin["privacy_level"] == 2
            assert span_pin["canonical_text"] == "beta"
            assert message_pin["index_text"] == "alpha beta gamma"
            assert span_pin["index_text"] == "beta"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_verbatim_pin_service_search_respects_conversation_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        connection = await runtime.open_connection()
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            service = VerbatimPinService(runtime)

            await users.create_user("usr_1")
            await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat A")
            await conversations.create_conversation("cnv_2", "usr_1", None, "coding_debug", "Chat B")
            await messages.create_message(
                "msg_1",
                "cnv_1",
                "user",
                1,
                "Scope locked phrase",
                3,
                {},
            )
            await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.MESSAGE,
                target_id="msg_1",
                privacy_level=0,
                created_by="usr_1",
            )

            same_conversation = await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="scope locked phrase",
                privacy_ceiling=0,
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                conversation_id="cnv_1",
                limit=10,
            )
            other_conversation = await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="scope locked phrase",
                privacy_ceiling=0,
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                conversation_id="cnv_2",
                limit=10,
            )

            assert [row["target_id"] for row in same_conversation] == ["msg_1"]
            assert other_conversation == []
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    ("space_id", "boundary_mode"),
    [
        ("space_vault", SpaceBoundaryMode.PRIVACY_VAULT),
        ("space_severed", SpaceBoundaryMode.SEVERANCE),
    ],
)
@pytest.mark.asyncio
async def test_verbatim_pin_service_enforces_space_boundaries_for_crud_and_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    space_id: str,
    boundary_mode: SpaceBoundaryMode,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        connection = await runtime.open_connection()
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            spaces = SpaceRepository(connection, runtime.clock)
            service = VerbatimPinService(runtime)

            await users.create_user("usr_1")
            await spaces.resolve_space(
                owner_user_id="usr_1",
                space_id=space_id,
                boundary_mode=boundary_mode,
                display_name=space_id,
                source_kind="explicit",
                source_id=space_id,
            )
            await conversations.create_conversation(
                "cnv_outside",
                "usr_1",
                None,
                "coding_debug",
                "Outside",
                platform_id="web",
            )
            await conversations.create_conversation(
                "cnv_inside",
                "usr_1",
                None,
                "coding_debug",
                "Inside",
                platform_id="web",
                active_space_id=space_id,
            )

            created_by_operation = {}
            for operation in ("get", "edit", "delete"):
                created_by_operation[operation] = await service.create_verbatim_pin(
                    connection,
                    user_id="usr_1",
                    scope=MemoryScope.GLOBAL_USER,
                    target_kind=VerbatimPinTargetKind.TEXT_SPAN,
                    target_id=f"{operation}_{space_id}",
                    conversation_id="cnv_inside",
                    platform_id="web",
                    active_space_id=space_id,
                    active_space_boundary_mode=boundary_mode,
                    canonical_text=f"{operation} boundary pin phrase for {space_id}",
                    index_text=f"{operation} boundary pin phrase",
                    privacy_level=0,
                    created_by="usr_1",
                )
                assert created_by_operation[operation]["space_id"] == space_id
                assert created_by_operation[operation]["space_boundary_mode"] == boundary_mode.value

            derived = await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.GLOBAL_USER,
                target_kind=VerbatimPinTargetKind.TEXT_SPAN,
                target_id=f"derived_{space_id}",
                conversation_id="cnv_inside",
                platform_id="web",
                canonical_text=f"derived boundary pin phrase for {space_id}",
                index_text="derived boundary pin phrase",
                privacy_level=0,
                created_by="usr_1",
            )
            assert derived["space_id"] == space_id
            assert derived["space_boundary_mode"] == boundary_mode.value
            expected_visible_ids = {
                row["id"] for row in created_by_operation.values()
            } | {derived["id"]}

            outside_context = {
                "conversation_id": "cnv_outside",
                "platform_id": "web",
                "active_space_id": None,
                "active_space_boundary_mode": None,
            }
            inside_context = {
                "conversation_id": "cnv_inside",
                "platform_id": "web",
                "active_space_id": space_id,
                "active_space_boundary_mode": boundary_mode,
            }

            outside_list = await service.list_verbatim_pins(
                connection,
                user_id="usr_1",
                **outside_context,
            )
            assert outside_list == []

            outside_search = await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="boundary pin phrase",
                privacy_ceiling=0,
                scope_filter=[MemoryScope.GLOBAL_USER],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                limit=10,
                **outside_context,
            )
            assert outside_search == []

            assert await service.get_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created_by_operation["get"]["id"],
                **outside_context,
            ) is None
            assert await service.update_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created_by_operation["edit"]["id"],
                canonical_text="outside edit must not land",
                **outside_context,
            ) is None
            assert await service.delete_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created_by_operation["delete"]["id"],
                **outside_context,
            ) is None

            inside_list = await service.list_verbatim_pins(
                connection,
                user_id="usr_1",
                **inside_context,
            )
            assert {row["id"] for row in inside_list} == expected_visible_ids

            inside_search = await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="boundary pin phrase",
                privacy_ceiling=0,
                scope_filter=[MemoryScope.GLOBAL_USER],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                limit=10,
                **inside_context,
            )
            assert {row["id"] for row in inside_search} == expected_visible_ids

            inside_get = await service.get_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created_by_operation["get"]["id"],
                **inside_context,
            )
            assert inside_get is not None
            assert inside_get["canonical_text"] == f"get boundary pin phrase for {space_id}"

            inside_edit = await service.update_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created_by_operation["edit"]["id"],
                canonical_text="inside edit is allowed",
                **inside_context,
            )
            assert inside_edit is not None
            assert inside_edit["canonical_text"] == "inside edit is allowed"

            inside_delete = await service.delete_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created_by_operation["delete"]["id"],
                **inside_context,
            )
            assert inside_delete is not None
            assert inside_delete["status"] == VerbatimPinStatus.DELETED.value
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_verbatim_pin_service_uses_safe_index_text_for_high_privacy_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        connection = await runtime.open_connection()
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            service = VerbatimPinService(runtime)

            await users.create_user("usr_1")
            await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Chat")

            created = await service.create_verbatim_pin(
                connection,
                user_id="usr_1",
                scope=MemoryScope.CONVERSATION,
                target_kind=VerbatimPinTargetKind.MESSAGE,
                target_id="msg_1",
                conversation_id="cnv_1",
                assistant_mode_id="coding_debug",
                canonical_text="Bank card PIN: 4512",
                index_text="bank card PIN",
                privacy_level=3,
                created_by="usr_1",
            )

            assert created["canonical_text"] == "Bank card PIN: 4512"
            assert created["index_text"] == "bank card PIN"

            matching_rows = await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="bank card PIN",
                privacy_ceiling=3,
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                conversation_id="cnv_1",
                limit=10,
                as_of="2026-03-30T12:00:00+00:00",
            )
            assert [row["id"] for row in matching_rows] == [created["id"]]

            assert await service.search_active_verbatim_pins(
                connection,
                user_id="usr_1",
                query="4512",
                privacy_ceiling=3,
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id="coding_debug",
                workspace_id=None,
                conversation_id="cnv_1",
                limit=10,
                as_of="2026-03-30T12:00:00+00:00",
            ) == []

            updated = await service.update_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created["id"],
                status=VerbatimPinStatus.ARCHIVED,
            )
            assert updated is not None
            assert updated["status"] == VerbatimPinStatus.ARCHIVED.value

            deleted = await service.delete_verbatim_pin(
                connection,
                user_id="usr_1",
                pin_id=created["id"],
            )
            assert deleted is not None
            assert deleted["status"] == VerbatimPinStatus.DELETED.value
        finally:
            await connection.close()
    finally:
        await runtime.close()
