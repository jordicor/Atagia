"""Tests for MCP server helper logic."""

from __future__ import annotations

# ruff: noqa: E402

import asyncio
from datetime import datetime, timezone
import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

mcp_available = pytest.importorskip("mcp", reason="mcp package not installed")

from atagia import Atagia
from atagia.core.clock import FrozenClock
from atagia.core.conversation_lifecycle_repository import (
    ConversationLifecycleRepository,
)
from atagia.core.initial_context_package_repository import (
    InitialContextPackageRepository,
)
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.space_repository import SpaceRepository
from atagia.core.transcript_rebuild_repository import TranscriptRebuildRepository
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.models.schemas_api import ContextResult
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageCoordinateSignature,
    InitialContextPackageKind,
    InitialContextPackagePolicySignature,
    initial_context_package_key_hash,
)
from atagia.models.schemas_jobs import (
    EXTRACT_STREAM_NAME,
    JobEnvelope,
    WORKER_GROUP_NAME,
)
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    SpaceBoundaryMode,
)
from atagia.mcp_server import (
    AtagiaContext,
    _add_memory_impl,
    _archive_conversation_impl,
    _close_conversation_impl,
    _delete_conversation_impl,
    _delete_memory_impl,
    _edit_memory_impl,
    _get_context_impl,
    _list_memories_impl,
    _search_memories_impl,
    _search_visible_mcp_memories,
    atagia_add_memory,
    atagia_list_memories,
    lifespan,
)
from atagia.services.errors import (
    ConversationNotFoundError,
    DeletionConfirmationError,
    MemoryNotFoundError,
    TranscriptRebuildInProgressError,
    TranscriptRebuildRemediationRequiredError,
)
from atagia.services.embeddings import EmbeddingIndex
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.initial_context_package_builder import (
    INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
)
from atagia.services.initial_context_package_keys import (
    build_initial_context_package_key,
)
from atagia.services.chat_support import default_operational_profile_snapshot


class _FakeMcpEngine:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, object] | None = None
        self.context_kwargs: dict[str, object] | None = None

    async def create_conversation(self, **kwargs):
        self.create_kwargs = kwargs
        return kwargs["conversation_id"]

    async def get_context(self, **kwargs):
        self.context_kwargs = kwargs
        return ContextResult(system_prompt="fake prompt")


from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from tests.extraction_payload_support import (
    is_memory_extraction_card_purpose,
    memory_extraction_card_output_from_payload,
)


async def _block_mcp_memory_access_for_selected_transcript(
    engine: Atagia,
    *,
    user_id: str,
    conversation_id: str,
    state: str,
) -> None:
    runtime = engine.runtime
    assert runtime is not None
    now = runtime.clock.now().isoformat()
    connection = await runtime.open_connection()
    try:
        await _insert_mcp_memory_access_blocker(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
            state=state,
            now=now,
        )
        await connection.commit()
    finally:
        await connection.close()


async def _insert_mcp_memory_access_blocker(
    connection,
    *,
    user_id: str,
    conversation_id: str,
    state: str,
    now: str,
) -> None:
    workflow_id = f"trw_mcp_{state}"
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
            workflow_id,
            f"op_mcp_{state}",
            user_id,
            conversation_id,
            1,
            f"hash_mcp_{state}",
            "replace",
            "[]",
            "[]",
            "[]",
            "[]",
            "[]",
            f"job_mcp_{state}",
            "remediation_required" if state == "remediation_required" else "aggregates",
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
        (
            user_id,
            conversation_id,
            1,
            f"hash_mcp_{state}",
            workflow_id,
            state,
            now,
        ),
    )


_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


def _is_need_detection_card_purpose(purpose: object) -> bool:
    value = str(purpose)
    return value.startswith("need_detection_") and value.endswith("_card")


class MCPProvider(LLMProvider):
    name = "mcp-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if _is_need_detection_card_purpose(purpose):
            outputs = {
                "need_detection_needs_card": "none",
                "need_detection_language_card": "en\nen",
                "need_detection_memory_card": "mixed",
                "need_detection_exact_card": "no",
                "need_detection_shape_card": "default",
                "need_detection_facets_card": "none",
                "need_detection_callback_card": "no",
                "need_detection_search_words_card": "retry loop",
                "need_detection_search_words_other_language_card": "none",
            }
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=outputs[purpose],
            )
        if purpose == "applicability_relevance_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} useful" for _memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "applicability_date_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} none" for _memory_id, score_key in candidate_keys
                ),
            )
        if is_memory_extraction_card_purpose(purpose):
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=memory_extraction_card_output_from_payload(
                    {"candidates": [], "nothing_durable": True},
                    purpose,
                ),
            )
        if purpose == "contract_projection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "signals": [],
                        "nothing_durable": True,
                    }
                ),
            )
        if purpose == "consequence_gate_card":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="no",
            )
        raise AssertionError(f"Unexpected LLM purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in MCP tests: {request.model}")


class TrackingEmbeddingIndex(EmbeddingIndex):
    def __init__(self) -> None:
        self.deleted_memory_ids: list[str] = []

    @property
    def vector_limit(self) -> int:
        return 1

    async def upsert(
        self, memory_id: str, text: str, metadata: dict[str, object]
    ) -> None:
        return None

    async def search(self, query: str, user_id: str, top_k: int):
        return []

    async def delete(self, memory_id: str) -> None:
        self.deleted_memory_ids.append(memory_id)


def _install_stub_client(
    monkeypatch: pytest.MonkeyPatch, provider: MCPProvider
) -> None:
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    # The MCP tests assert the full retrieval pipeline runs (including need
    # detection), so disable the small-corpus shortcut for the duration of
    # the test.
    monkeypatch.setenv("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0")


def _normal_operational_profile_token(engine: Atagia) -> str:
    if engine.runtime is None:
        raise AssertionError("Engine runtime should be initialized")
    return default_operational_profile_snapshot(
        loader=engine.runtime.operational_profile_loader,
        settings=engine.runtime.settings,
    ).token


async def _active_cache_identity(engine: Atagia, user_id: str) -> tuple[str, int, int]:
    if engine.runtime is None:
        raise AssertionError("Engine runtime should be initialized")
    connection = await engine.runtime.open_connection()
    try:
        identity = await UserLifecycleRepository(
            connection,
            engine.runtime.clock,
        ).get_active_identity(user_id)
    finally:
        await connection.close()
    if identity is None:
        raise AssertionError("Active user lifecycle identity should exist")
    return (
        identity.lifecycle_epoch,
        identity.cache_revision,
        identity.derivation_revision,
    )


async def _seed_memory(
    engine: Atagia,
    *,
    memory_id: str,
    text: str,
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    privacy_level: int = 0,
    scope: MemoryScope = MemoryScope.CONVERSATION,
    conversation_id: str | None = "cnv_1",
    platform_id: str = "web",
    space_id: str | None = None,
    space_boundary_mode: SpaceBoundaryMode | None = None,
    embodiment_id: str | None = None,
    realm_id: str | None = None,
) -> None:
    runtime = engine.runtime
    if runtime is None:
        raise AssertionError(
            "Engine runtime should be initialized before seeding memories"
        )
    connection = await runtime.open_connection()
    try:
        memories = MemoryObjectRepository(connection, runtime.clock)
        scope_canonical = {
            MemoryScope.CONVERSATION: MemoryScope.CHAT.value,
            MemoryScope.EPHEMERAL_SESSION: MemoryScope.CHAT.value,
            MemoryScope.WORKSPACE: MemoryScope.CHARACTER.value,
            MemoryScope.GLOBAL_USER: MemoryScope.USER.value,
        }.get(scope)
        await memories.create_memory_object(
            user_id="usr_1",
            conversation_id=conversation_id,
            assistant_mode_id="coding_debug",
            object_type=object_type,
            scope=scope,
            canonical_text=text,
            source_kind=MemorySourceKind.VERBATIM,
            confidence=0.9,
            privacy_level=privacy_level,
            status=status,
            memory_id=memory_id,
            platform_id=platform_id,
            scope_canonical=scope_canonical,
            space_id=space_id,
            space_boundary_mode=space_boundary_mode.value
            if space_boundary_mode is not None
            else None,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_mcp_get_context_passes_full_coordinate_fields() -> None:
    engine = _FakeMcpEngine()

    payload = json.loads(
        await _get_context_impl(
            engine,  # type: ignore[arg-type]
            "usr_1",
            "mcp",
            "Coordinate-aware context please.",
            conversation_id="cnv_1",
            mode="coding_debug",
            user_persona_id="persona_mcp",
            character_id="char_mcp",
            active_presence_id="presence_mcp",
            mind_id="mind_mcp",
            mind_topology="ojocentauri",
            embodiment_id="body_mcp",
            realm_id="realm_mcp",
            space_id="space_mcp",
            incognito=True,
        )
    )

    assert payload["conversation_id"] == "cnv_1"
    assert engine.create_kwargs is not None
    assert engine.context_kwargs is not None
    for key, expected in {
        "active_presence_id": "presence_mcp",
        "mind_id": "mind_mcp",
        "mind_topology": "ojocentauri",
        "embodiment_id": "body_mcp",
        "realm_id": "realm_mcp",
        "space_id": "space_mcp",
    }.items():
        assert engine.create_kwargs[key] == expected
        assert engine.context_kwargs[key] == expected


@pytest.mark.asyncio
async def test_mcp_get_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        first = json.loads(
            await _get_context_impl(
                engine,
                "usr_1",
                "mcp",
                "Please help me debug this retry loop.",
                conversation_id="cnv_1",
                mode="coding_debug",
                user_persona_id="persona_mcp",
                character_id="char_mcp",
            )
        )
        second = json.loads(
            await _get_context_impl(
                engine,
                "usr_1",
                "mcp",
                "continue",
                conversation_id="cnv_1",
                mode="coding_debug",
                user_persona_id="persona_mcp",
                character_id="char_mcp",
            )
        )

        assert first["system_prompt"]
        assert first["conversation_id"] == "cnv_1"
        assert second["conversation_id"] == "cnv_1"
        # The first turn is an empty-clean cold start (no stored memories,
        # contracts, or prior messages), so the pipeline composes an empty
        # context without invoking need detection. The second turn has a prior
        # persisted message, so it runs the full pipeline and issues the
        # parallel need-detection card set.
        assert (
            sum(
                _is_need_detection_card_purpose(request.metadata.get("purpose"))
                for request in provider.requests
            )
            == 8
        )
        (
            lifecycle_epoch,
            cache_revision,
            derivation_revision,
        ) = await _active_cache_identity(engine, "usr_1")
        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=None,
            active_presence_id="char_mcp",
            operational_profile_token=_normal_operational_profile_token(engine),
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
        )
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None
        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(
                connection, runtime.clock
            ).get_conversation(
                "cnv_1",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["user_persona_id"] == "persona_mcp"
            assert conversation["character_id"] == "char_mcp"
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.parametrize(
    ("state", "error_type", "error_message"),
    [
        (
            "rebuilding",
            TranscriptRebuildInProgressError,
            "Selected transcript rebuild is still in progress",
        ),
        (
            "remediation_required",
            TranscriptRebuildRemediationRequiredError,
            "Selected transcript rebuild requires remediation before memory access",
        ),
    ],
)
@pytest.mark.asyncio
async def test_mcp_memory_tools_fail_closed_during_selected_transcript_rebuild(
    tmp_path: Path,
    state: str,
    error_type: type[Exception],
    error_message: str,
) -> None:
    engine = Atagia(
        db_path=tmp_path / f"mcp-selected-{state}.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    await engine.setup()
    try:
        await engine.create_user("usr_selected")
        await engine.create_conversation(
            "usr_selected",
            "cnv_selected",
            platform_id="mcp",
        )
        await _block_mcp_memory_access_for_selected_transcript(
            engine,
            user_id="usr_selected",
            conversation_id="cnv_selected",
            state=state,
        )
        lifespan_context = AtagiaContext(
            engine=engine,
            user_id="usr_selected",
            platform_id="mcp",
            conversation_id="cnv_selected",
        )
        ctx = SimpleNamespace(
            request_context=SimpleNamespace(lifespan_context=lifespan_context)
        )

        expected_error = f"Error: {error_message}"
        with pytest.raises(
            error_type,
            match=f"^{re.escape(error_message)}$",
        ):
            await _list_memories_impl(
                engine,
                "usr_selected",
                conversation_id="cnv_selected",
                platform_id="mcp",
            )
        assert await atagia_list_memories(ctx=ctx) == expected_error
        assert (
            await atagia_add_memory(
                "This MCP write must not become visible.",
                ctx=ctx,
            )
            == expected_error
        )

        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation("cnv_selected", "usr_selected")
        finally:
            await connection.close()
        assert messages == []
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_add_memory_selection_wins_before_message_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "mcp-add-selection-first.db"
    engine = Atagia(
        db_path=database_path,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    competing_engine = Atagia(
        db_path=database_path,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    await engine.setup()
    await engine.create_user("usr_selected")
    await engine.create_conversation(
        "usr_selected",
        "cnv_selected",
        platform_id="mcp",
    )
    await competing_engine.setup()
    snapshot_captured = asyncio.Event()
    selection_committed = asyncio.Event()
    original_capture = TranscriptRebuildRepository.capture_user_availability_snapshot

    async def pause_after_snapshot(repository, user_id, **kwargs):
        snapshot = await original_capture(repository, user_id, **kwargs)
        if user_id == "usr_selected" and not snapshot_captured.is_set():
            snapshot_captured.set()
            await selection_committed.wait()
        return snapshot

    monkeypatch.setattr(
        TranscriptRebuildRepository,
        "capture_user_availability_snapshot",
        pause_after_snapshot,
    )
    try:
        add_task = asyncio.create_task(
            _add_memory_impl(
                engine,
                "usr_selected",
                "mcp",
                "This message must lose to the selected transcript.",
                conversation_id="cnv_selected",
            )
        )
        await asyncio.wait_for(snapshot_captured.wait(), timeout=2.0)
        await _block_mcp_memory_access_for_selected_transcript(
            competing_engine,
            user_id="usr_selected",
            conversation_id="cnv_selected",
            state="rebuilding",
        )
        selection_committed.set()
        with pytest.raises(
            TranscriptRebuildInProgressError,
            match="Selected transcript rebuild is still in progress",
        ):
            await add_task

        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            message_count = await connection.execute_fetchall(
                "SELECT id FROM messages WHERE conversation_id = ?",
                ("cnv_selected",),
            )
            job_count = await connection.execute_fetchall(
                "SELECT job_id FROM worker_job_runs WHERE user_id = ?",
                ("usr_selected",),
            )
        finally:
            await connection.close()
        assert message_count == []
        assert job_count == []
    finally:
        selection_committed.set()
        await competing_engine.close()
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_add_memory_message_and_jobs_commit_before_waiting_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "mcp-add-atomic-jobs.db"
    engine = Atagia(
        db_path=database_path,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    competing_engine = Atagia(
        db_path=database_path,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    await engine.setup()
    await engine.create_user("usr_selected")
    await engine.create_conversation(
        "usr_selected",
        "cnv_selected",
        platform_id="mcp",
    )
    await competing_engine.setup()

    from atagia import mcp_server

    message_inserted = asyncio.Event()
    allow_job_records = asyncio.Event()
    original_enqueue = mcp_server.enqueue_message_jobs

    async def pause_before_job_records(**kwargs):
        message_inserted.set()
        await allow_job_records.wait()
        return await original_enqueue(**kwargs)

    monkeypatch.setattr(
        mcp_server,
        "enqueue_message_jobs",
        pause_before_job_records,
    )
    writer_attempted = asyncio.Event()
    writer_acquired = asyncio.Event()

    async def install_waiting_selection() -> None:
        runtime = competing_engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            writer_attempted.set()
            await connection.execute("BEGIN IMMEDIATE")
            writer_acquired.set()
            await _insert_mcp_memory_access_blocker(
                connection,
                user_id="usr_selected",
                conversation_id="cnv_selected",
                state="rebuilding",
                now=runtime.clock.now().isoformat(),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
        finally:
            await connection.close()

    try:
        add_task = asyncio.create_task(
            _add_memory_impl(
                engine,
                "usr_selected",
                "mcp",
                "The message and its jobs are one durable unit.",
                conversation_id="cnv_selected",
            )
        )
        await asyncio.wait_for(message_inserted.wait(), timeout=2.0)
        selection_task = asyncio.create_task(install_waiting_selection())
        await asyncio.wait_for(writer_attempted.wait(), timeout=2.0)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(writer_acquired.wait(), timeout=0.05)

        competing_runtime = competing_engine.runtime
        assert competing_runtime is not None
        observer = await competing_runtime.open_connection()
        try:
            visible_messages = await observer.execute_fetchall(
                "SELECT id FROM messages WHERE conversation_id = ?",
                ("cnv_selected",),
            )
            visible_jobs = await observer.execute_fetchall(
                "SELECT job_id FROM worker_job_runs WHERE user_id = ?",
                ("usr_selected",),
            )
        finally:
            await observer.close()
        assert visible_messages == []
        assert visible_jobs == []

        allow_job_records.set()
        confirmation, _ = await asyncio.gather(add_task, selection_task)
        assert "Stored memory candidate message" in confirmation
        assert writer_acquired.is_set()

        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            committed_messages = await connection.execute_fetchall(
                "SELECT id FROM messages WHERE conversation_id = ?",
                ("cnv_selected",),
            )
            committed_jobs = await connection.execute_fetchall(
                """
                SELECT job_type
                FROM worker_job_runs
                WHERE user_id = ? AND conversation_id = ?
                """,
                ("usr_selected", "cnv_selected"),
            )
            selection = await TranscriptRebuildRepository(
                connection,
                runtime.clock,
            ).get_blocking_selection("usr_selected")
        finally:
            await connection.close()
        assert len(committed_messages) == 1
        assert {str(row["job_type"]) for row in committed_jobs} >= {
            "extract_memory_candidates",
            "project_contract",
        }
        assert selection is not None
        assert selection["state"] == "rebuilding"
    finally:
        allow_job_records.set()
        await competing_engine.close()
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_list_holds_namespace_and_rows_in_one_sqlite_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = Atagia(
        db_path=tmp_path / "mcp-read-snapshot-race.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_mcp_read_race",
            text="Stale selected branch memory",
        )
        runtime = engine.runtime
        assert runtime is not None

        from atagia import mcp_server

        original_list = mcp_server._list_visible_mcp_memories

        writer_started = asyncio.Event()
        writer_acquired = asyncio.Event()
        replacement_task: asyncio.Task[None] | None = None

        async def list_while_replacement_waits(*args, **kwargs):
            nonlocal replacement_task
            rows = await original_list(*args, **kwargs)

            async def complete_replacement() -> None:
                writer = await runtime.open_connection()
                try:
                    writer_started.set()
                    await writer.execute("BEGIN IMMEDIATE")
                    writer_acquired.set()
                    await writer.execute(
                        "DELETE FROM memory_objects WHERE id = ? AND user_id = ?",
                        ("mem_mcp_read_race", "usr_1"),
                    )
                    await writer.execute(
                        """
                        UPDATE user_lifecycles
                        SET derivation_revision = derivation_revision + 1
                        WHERE user_id = ?
                        """,
                        ("usr_1",),
                    )
                    await writer.commit()
                finally:
                    await writer.close()

            replacement_task = asyncio.create_task(complete_replacement())
            await asyncio.wait_for(writer_started.wait(), timeout=2.0)
            await asyncio.sleep(0.05)
            assert not writer_acquired.is_set()
            return rows

        monkeypatch.setattr(
            mcp_server,
            "_list_visible_mcp_memories",
            list_while_replacement_waits,
        )
        result = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_1",
                platform_id="web",
            )
        )
        assert {memory["id"] for memory in result} == {"mem_mcp_read_race"}
        assert replacement_task is not None
        await asyncio.wait_for(replacement_task, timeout=2.0)
        assert writer_acquired.is_set()
        monkeypatch.setattr(
            mcp_server,
            "_list_visible_mcp_memories",
            original_list,
        )
        assert (
            json.loads(
                await _list_memories_impl(
                    engine,
                    "usr_1",
                    conversation_id="cnv_1",
                    platform_id="web",
                )
            )
            == []
        )
    finally:
        await engine.close()


@pytest.mark.parametrize(
    "operation",
    (
        "edit_memory",
        "archive_memory",
        "hard_delete_memory",
        "close_conversation",
        "archive_conversation",
        "delete_conversation",
    ),
)
@pytest.mark.asyncio
async def test_mcp_mutations_reject_namespace_change_after_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    engine = Atagia(
        db_path=tmp_path / f"mcp-namespace-race-{operation}.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    await engine.setup()
    try:
        await engine.create_user("usr_1")
        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            spaces = SpaceRepository(connection, runtime.clock)
            for space_id in ("space_old", "space_new"):
                await spaces.resolve_space(
                    owner_user_id="usr_1",
                    space_id=space_id,
                    boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT,
                    display_name=space_id,
                    source_kind="explicit",
                    source_id=space_id,
                )
        finally:
            await connection.close()
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="web",
            space_id="space_old",
        )
        await _seed_memory(
            engine,
            memory_id="mem_namespace_race",
            text="Original namespace-bound memory",
            scope=MemoryScope.GLOBAL_USER,
            conversation_id=None,
            platform_id="web",
            space_id="space_old",
            space_boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT,
        )

        from atagia import mcp_server

        original_snapshot = mcp_server._mcp_namespace_snapshot

        async def snapshot_then_move_namespace(*args, **kwargs):
            snapshot = await original_snapshot(*args, **kwargs)
            writer = await runtime.open_connection()
            try:
                await writer.execute("BEGIN IMMEDIATE")
                await writer.execute(
                    """
                    UPDATE conversations
                    SET active_space_id = ?, updated_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (
                        "space_new",
                        runtime.clock.now().isoformat(),
                        "cnv_1",
                        "usr_1",
                    ),
                )
                await writer.execute(
                    """
                    UPDATE user_lifecycles
                    SET derivation_revision = derivation_revision + 1,
                        updated_at = ?
                    WHERE user_id = ?
                    """,
                    (runtime.clock.now().isoformat(), "usr_1"),
                )
                await writer.commit()
            finally:
                await writer.close()
            return snapshot

        monkeypatch.setattr(
            mcp_server,
            "_mcp_namespace_snapshot",
            snapshot_then_move_namespace,
        )

        with pytest.raises(
            ConversationNotFoundError,
            match="Conversation namespace changed",
        ):
            if operation == "edit_memory":
                await _edit_memory_impl(
                    engine,
                    "usr_1",
                    "mem_namespace_race",
                    "Unauthorized stale-namespace edit",
                    conversation_id="cnv_1",
                    platform_id="web",
                )
            elif operation in {"archive_memory", "hard_delete_memory"}:
                hard = operation == "hard_delete_memory"
                await _delete_memory_impl(
                    engine,
                    "usr_1",
                    "mem_namespace_race",
                    hard=hard,
                    confirmation="HARD_DELETE_MEMORY" if hard else None,
                    conversation_id="cnv_1",
                    platform_id="web",
                )
            elif operation == "close_conversation":
                await _close_conversation_impl(
                    engine,
                    "usr_1",
                    "cnv_1",
                    platform_id="web",
                )
            elif operation == "archive_conversation":
                await _archive_conversation_impl(
                    engine,
                    "usr_1",
                    "cnv_1",
                    platform_id="web",
                )
            else:
                await _delete_conversation_impl(
                    engine,
                    "usr_1",
                    "cnv_1",
                    platform_id="web",
                    confirmation="DELETE_CONVERSATION",
                )

        inspection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(
                inspection,
                runtime.clock,
            ).get_conversation("cnv_1", "usr_1")
            memory = await MemoryObjectRepository(
                inspection,
                runtime.clock,
            ).get_memory_object("mem_namespace_race", "usr_1")
        finally:
            await inspection.close()
        assert conversation is not None
        assert conversation["active_space_id"] == "space_new"
        assert conversation["status"] == "active"
        assert memory is not None
        assert memory["canonical_text"] == "Original namespace-bound memory"
        assert memory["status"] == MemoryStatus.ACTIVE.value
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("aba_kind", ("conversation", "user"))
async def test_mcp_stale_mutation_rejects_recreated_conversation_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aba_kind: str,
) -> None:
    engine = Atagia(
        db_path=tmp_path / f"mcp-{aba_kind}-aba.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        runtime = engine.runtime
        assert runtime is not None

        from atagia import mcp_server

        original_snapshot = mcp_server._mcp_namespace_snapshot
        old_epoch: str | None = None
        old_user_epoch: str | None = None

        async def snapshot_then_recreate(*args, **kwargs):
            nonlocal old_epoch, old_user_epoch
            snapshot = await original_snapshot(*args, **kwargs)
            old_epoch = snapshot.conversation_lifecycle_epoch
            old_user_epoch = snapshot.user_lifecycle_epoch
            writer = await runtime.open_connection()
            try:
                await writer.execute("BEGIN IMMEDIATE")
                await writer.execute(
                    "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                    ("cnv_1", "usr_1"),
                )
                if aba_kind == "user":
                    await writer.execute(
                        "DELETE FROM users WHERE id = ?",
                        ("usr_1",),
                    )
                    await writer.execute(
                        "DELETE FROM user_lifecycles WHERE user_id = ?",
                        ("usr_1",),
                    )
                await writer.commit()
            finally:
                await writer.close()
            if aba_kind == "user":
                await engine.create_user("usr_1")
            await engine.create_conversation(
                "usr_1",
                "cnv_1",
                assistant_mode_id="coding_debug",
                platform_id="web",
            )
            return snapshot

        monkeypatch.setattr(
            mcp_server,
            "_mcp_namespace_snapshot",
            snapshot_then_recreate,
        )
        with pytest.raises(
            ConversationNotFoundError,
            match="Conversation namespace changed",
        ):
            await _close_conversation_impl(
                engine,
                "usr_1",
                "cnv_1",
                platform_id="web",
            )

        inspection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(
                inspection,
                runtime.clock,
            ).get_conversation("cnv_1", "usr_1")
            identity = await ConversationLifecycleRepository(
                inspection,
                runtime.clock,
            ).get_identity(user_id="usr_1", conversation_id="cnv_1")
            user_identity = await UserLifecycleRepository(
                inspection,
                runtime.clock,
            ).get_active_identity("usr_1")
        finally:
            await inspection.close()
        assert conversation is not None
        assert conversation["status"] == "active"
        assert identity is not None
        assert old_epoch is not None
        assert identity.lifecycle_epoch != old_epoch
        assert user_identity is not None
        assert old_user_epoch is not None
        if aba_kind == "user":
            assert user_identity.lifecycle_epoch != old_user_epoch
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_lifespan_reads_embodiment_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_DB_PATH", str(tmp_path / "atagia-mcp-env.db"))
    monkeypatch.setenv("ATAGIA_USER_ID", "usr_1")
    monkeypatch.setenv("ATAGIA_PLATFORM_ID", "mcp")
    monkeypatch.setenv("ATAGIA_EMBODIMENT_ID", "body_env")
    monkeypatch.setenv("ATAGIA_REALM_ID", "realm_env")
    monkeypatch.setenv("ATAGIA_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ATAGIA_LLM_FORCED_GLOBAL_MODEL", "openai/test-model")

    async with lifespan(None) as context:
        assert context.user_id == "usr_1"
        assert context.platform_id == "mcp"
        assert context.embodiment_id == "body_env"
        assert context.realm_id == "realm_env"


@pytest.mark.asyncio
async def test_mcp_context_and_add_memory_propagate_embodiment_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await _get_context_impl(
            engine,
            "usr_1",
            "mcp",
            "Please help me debug this retry loop.",
            conversation_id="cnv_body",
            mode="coding_debug",
            user_persona_id="persona_mcp",
            character_id="char_mcp",
            embodiment_id="body_mcp",
        )
        await _add_memory_impl(
            engine,
            "usr_1",
            "mcp",
            "Please remember that this chat is on the headset.",
            conversation_id="cnv_body",
            user_persona_id="persona_mcp",
            character_id="char_mcp",
            embodiment_id="body_mcp",
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(
                connection, runtime.clock
            ).get_conversation(
                "cnv_body",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["active_embodiment_id"] == "body_mcp"
            messages = await MessageRepository(connection, runtime.clock).get_messages(
                "cnv_body",
                "usr_1",
                limit=10,
                offset=0,
            )
            assert len(messages) >= 2
            assert {message["active_embodiment_id"] for message in messages} == {
                "body_mcp"
            }
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_context_and_add_memory_propagate_realm_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await _get_context_impl(
            engine,
            "usr_1",
            "mcp",
            "Please help me debug this retry loop.",
            conversation_id="cnv_realm",
            mode="coding_debug",
            user_persona_id="persona_mcp",
            character_id="char_mcp",
            realm_id="realm_mcp",
        )
        await _add_memory_impl(
            engine,
            "usr_1",
            "mcp",
            "Please remember that this chat is in a story realm.",
            conversation_id="cnv_realm",
            user_persona_id="persona_mcp",
            character_id="char_mcp",
            realm_id="realm_mcp",
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(
                connection, runtime.clock
            ).get_conversation(
                "cnv_realm",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["active_realm_id"] == "realm_mcp"
            messages = await MessageRepository(connection, runtime.clock).get_messages(
                "cnv_realm",
                "usr_1",
                limit=10,
                offset=0,
            )
            assert len(messages) >= 2
            assert {message["active_realm_id"] for message in messages} == {"realm_mcp"}
        finally:
            await connection.close()
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_memory_tools_can_apply_env_embodiment_to_existing_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_env",
            assistant_mode_id="coding_debug",
            platform_id="mcp",
        )
        await _seed_memory(
            engine,
            memory_id="mem_body_env",
            text="environmentbodytoken memory belongs to the headset",
            scope=MemoryScope.GLOBAL_USER,
            conversation_id=None,
            platform_id="mcp",
            embodiment_id="body_env",
        )

        before_env = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_env",
                platform_id="mcp",
            )
        )
        with_env = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_env",
                platform_id="mcp",
                embodiment_id="body_env",
            )
        )
        search_with_env = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "environmentbodytoken",
                conversation_id="cnv_env",
                platform_id="mcp",
                embodiment_id="body_env",
            )
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(
                connection, runtime.clock
            ).get_conversation(
                "cnv_env",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["active_embodiment_id"] == "body_env"
        finally:
            await connection.close()

        assert {memory["id"] for memory in before_env} == set()
        assert {memory["id"] for memory in with_env} == {"mem_body_env"}
        assert {memory["id"] for memory in search_with_env} == {"mem_body_env"}
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_memory_tools_can_apply_env_realm_to_existing_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_realm_env",
            assistant_mode_id="coding_debug",
            platform_id="mcp",
        )
        await _seed_memory(
            engine,
            memory_id="mem_realm_env",
            text="environmentrealmtoken memory belongs to the story realm",
            scope=MemoryScope.GLOBAL_USER,
            conversation_id=None,
            platform_id="mcp",
            realm_id="realm_env",
        )

        before_env = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_realm_env",
                platform_id="mcp",
            )
        )
        with_env = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_realm_env",
                platform_id="mcp",
                realm_id="realm_env",
            )
        )
        search_with_env = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "environmentrealmtoken",
                conversation_id="cnv_realm_env",
                platform_id="mcp",
                realm_id="realm_env",
            )
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            conversation = await ConversationRepository(
                connection, runtime.clock
            ).get_conversation(
                "cnv_realm_env",
                "usr_1",
            )
            assert conversation is not None
            assert conversation["active_realm_id"] == "realm_env"
        finally:
            await connection.close()

        assert {memory["id"] for memory in before_env} == set()
        assert {memory["id"] for memory in with_env} == {"mem_realm_env"}
        assert {memory["id"] for memory in search_with_env} == {"mem_realm_env"}
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_search_memories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_1",
            text="retry loop websocket backoff",
        )
        await _seed_memory(
            engine,
            memory_id="mem_archived",
            text="archived retry memory",
        )
        await _seed_memory(
            engine,
            memory_id="mem_pending",
            text="pending retry credential",
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        await _seed_memory(
            engine,
            memory_id="mem_private",
            text="private retry credential",
            privacy_level=2,
        )
        await _seed_memory(
            engine,
            memory_id="mem_declined",
            text="declined retry credential",
            status=MemoryStatus.DECLINED,
        )
        await _delete_memory_impl(
            engine,
            "usr_1",
            "mem_archived",
            conversation_id="cnv_1",
            platform_id="web",
        )

        results = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "retry",
                limit=10,
                conversation_id="cnv_1",
                platform_id="web",
            )
        )

        assert results
        assert results[0]["id"] == "mem_1"
        assert "retry loop websocket backoff" in results[0]["text"]
        assert all(result["id"] != "mem_archived" for result in results)
        assert all(result["id"] != "mem_pending" for result in results)
        assert all(result["id"] != "mem_private" for result in results)
        assert all(result["id"] != "mem_declined" for result in results)
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_search_sanitizes_arbitrary_fts_grammar_and_partitions_by_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp-fts-safety.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_cpp",
            text="C++ systems notes with foo bar and retry guidance",
        )
        await _seed_memory(
            engine,
            memory_id="mem_unicode",
            text="Unicode retrieval notes in 日本語 and Español",
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            await users.create_user("usr_other")
            await conversations.create_conversation(
                "cnv_other",
                "usr_other",
                None,
                "coding_debug",
                "Other user",
                platform_id="web",
            )
            await memories.create_memory_object(
                user_id="usr_other",
                conversation_id="cnv_other",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.CONVERSATION,
                canonical_text="C++ foo bar retry 日本語 private other-user memory",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                status=MemoryStatus.ACTIVE,
                memory_id="mem_other_user",
                platform_id="web",
                scope_canonical=MemoryScope.CHAT.value,
            )
        finally:
            await connection.close()

        expected_hits = {
            "C++": "mem_cpp",
            "foo:bar": "mem_cpp",
            '"retry': "mem_cpp",
            "日本語": "mem_unicode",
        }
        for query, expected_id in expected_hits.items():
            results = json.loads(
                await _search_memories_impl(
                    engine,
                    "usr_1",
                    query,
                    conversation_id="cnv_1",
                    platform_id="web",
                )
            )
            assert expected_id in {item["id"] for item in results}
            assert "mem_other_user" not in {item["id"] for item in results}

        grammar_fuzz = [
            "***",
            "AND OR NOT",
            "() [] {}",
            "foo NEAR/ bar",
            "'unmatched \"quotes",
            "column:value^10~2",
            "C++ && foo:bar || !retry",
            "🧠🚀✨",
            "مرحبا:世界 + привет*",
            "\x00\x01\n\t",
        ]
        for query in grammar_fuzz:
            results = json.loads(
                await _search_memories_impl(
                    engine,
                    "usr_1",
                    query,
                    conversation_id="cnv_1",
                    platform_id="web",
                )
            )
            assert isinstance(results, list)
            assert all(item["id"] != "mem_other_user" for item in results)
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_search_translates_residual_sqlite_fts_errors() -> None:
    class FailingConnection:
        async def execute(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("fts5: syntax error near quote")

    with pytest.raises(ValueError, match="Memory search query could not be processed"):
        await _search_visible_mcp_memories(
            FailingConnection(),
            user_id="usr_1",
            query="retry",
            limit=10,
            conversation_id="cnv_1",
            user_persona_id=None,
            platform_id="web",
            character_id=None,
            incognito=False,
            remember_across_chats=True,
            remember_across_devices=True,
            active_space_id=None,
            active_space_boundary_mode=None,
            active_mind_id=None,
            mind_topology=None,
            active_embodiment_id=None,
            active_realm_id=None,
        )


@pytest.mark.asyncio
async def test_mcp_add_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        engine.runtime.clock = FrozenClock(
            datetime(2026, 3, 31, 4, 0, tzinfo=timezone.utc)
        )
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="mcp",
        )
        (
            lifecycle_epoch,
            cache_revision,
            derivation_revision,
        ) = await _active_cache_identity(engine, "usr_1")
        cache_key = ContextCacheService.build_cache_key(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            workspace_id=None,
            active_presence_id="default_assistant",
            operational_profile_token=_normal_operational_profile_token(engine),
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
        )
        await engine.runtime.storage_backend.set_context_view(
            cache_key,
            {"user_id": "usr_1", "conversation_id": "cnv_1"},
            ttl_seconds=60,
        )
        await engine.runtime.storage_backend.set_context_view(
            "ctx:other",
            {"user_id": "usr_2", "conversation_id": "cnv_2"},
            ttl_seconds=60,
        )
        operational_profile = default_operational_profile_snapshot(
            loader=engine.runtime.operational_profile_loader,
            settings=engine.runtime.settings,
        )
        policy_signature = InitialContextPackagePolicySignature(
            effective_policy_hash="policy-mcp-test",
            policy_prompt_hash="prompt-mcp-test",
            privacy_enforcement="enforce",
        )
        coordinate_signature = InitialContextPackageCoordinateSignature(
            coordinate_signature_hash="coord-mcp-test",
            complete=True,
        )
        package_key = build_initial_context_package_key(
            version=INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
            package_kind=InitialContextPackageKind.CONVERSATION,
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            subject_json={
                "user_persona_id": None,
                "platform_id": "mcp",
                "character_id": None,
                "workspace_id": None,
                "assistant_mode_id": "coding_debug",
                "mode": "coding_debug",
            },
            policy_signature=policy_signature,
            coordinate_signature=coordinate_signature,
            operational_profile=operational_profile,
        )
        package_hash = initial_context_package_key_hash(package_key)
        package_connection = await engine.runtime.open_connection()
        try:
            await InitialContextPackageRepository(
                package_connection,
                engine.runtime.clock,
            ).upsert_package(
                package_kind=InitialContextPackageKind.CONVERSATION,
                version=INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
                key_json=package_key,
                policy_signature_json=policy_signature,
                coordinate_signature_json=coordinate_signature,
                blocks_json={
                    "prepared_memory_profile_block": "Existing MCP prepared context.",
                    "source_counts": {"profile_items": 1},
                },
            )
        finally:
            await package_connection.close()

        confirmation = await _add_memory_impl(
            engine,
            "usr_1",
            "mcp",
            "Please remember that the retry loop needs a backoff.",
            conversation_id="cnv_1",
        )

        assert "Stored memory candidate message" in confirmation
        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            messages = MessageRepository(connection, runtime.clock)
            stored_messages = await messages.get_messages(
                "cnv_1", "usr_1", limit=10, offset=0
            )
            assert stored_messages[-1]["role"] == "user"
            assert (
                stored_messages[-1]["text"]
                == "Please remember that the retry loop needs a backoff."
            )
            assert stored_messages[-1]["occurred_at"] == "2026-03-31T04:00:00+00:00"
            package_status = await InitialContextPackageRepository(
                connection,
                runtime.clock,
            ).read_by_key_hash(
                user_id="usr_1",
                package_key_hash=package_hash,
            )
            assert package_status.status == "stale"
        finally:
            await connection.close()
        assert await engine.runtime.storage_backend.get_context_view(cache_key) is None
        assert await engine.runtime.storage_backend.get_context_view("ctx:other") == {
            "user_id": "usr_2",
            "conversation_id": "cnv_2",
        }
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_add_memory_carries_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="mcp",
        )

        await _add_memory_impl(
            engine,
            "usr_1",
            "mcp",
            "Please remember that offline mode should keep answers short.",
            conversation_id="cnv_1",
            operational_profile="offline",
        )

        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        messages = await runtime.storage_backend.stream_read(
            EXTRACT_STREAM_NAME,
            WORKER_GROUP_NAME,
            "test-consumer",
            count=1,
            block_ms=0,
        )
        if messages:
            envelope = JobEnvelope.model_validate(messages[0].payload)
            assert envelope.operational_profile is not None
            assert envelope.operational_profile.profile_id == "offline"
        else:
            connection = await runtime.open_connection()
            try:
                cursor = await connection.execute(
                    """
                    SELECT metadata_json
                    FROM worker_job_runs
                    WHERE user_id = ?
                      AND conversation_id = ?
                      AND job_type = ?
                    """,
                    ("usr_1", "cnv_1", "extract_memory_candidates"),
                )
                rows = await cursor.fetchall()
            finally:
                await connection.close()
            assert any(
                json.loads(row["metadata_json"]).get("operational_profile") == "offline"
                for row in rows
            )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_delete_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        tracking_embeddings = TrackingEmbeddingIndex()
        if engine.runtime is None:
            raise AssertionError("Engine runtime should be initialized")
        engine.runtime.embedding_index = tracking_embeddings
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_1",
            text="retry loop websocket backoff",
        )
        await engine.runtime.storage_backend.set_context_view(
            "ctx:1",
            {"user_id": "usr_1", "conversation_id": "cnv_1"},
            ttl_seconds=60,
        )
        await engine.runtime.storage_backend.set_context_view(
            "ctx:2",
            {"user_id": "usr_1", "conversation_id": "cnv_2"},
            ttl_seconds=60,
        )
        await engine.runtime.storage_backend.set_context_view(
            "ctx:3",
            {"user_id": "usr_2", "conversation_id": "cnv_3"},
            ttl_seconds=60,
        )

        confirmation = await _delete_memory_impl(
            engine,
            "usr_1",
            "mem_1",
            conversation_id="cnv_1",
            platform_id="web",
        )

        assert confirmation == "Archived memory mem_1."
        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should remain initialized")
        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            memory = await memories.get_memory_object("mem_1", "usr_1")
            assert memory is not None
            assert memory["status"] == "archived"
        finally:
            await connection.close()
        assert tracking_embeddings.deleted_memory_ids == ["mem_1"]
        assert await engine.runtime.storage_backend.get_context_view("ctx:1") is None
        assert await engine.runtime.storage_backend.get_context_view("ctx:2") is None
        assert await engine.runtime.storage_backend.get_context_view("ctx:3") == {
            "user_id": "usr_2",
            "conversation_id": "cnv_3",
        }
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_destructive_deletes_require_explicit_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_1",
            text="destructive delete confirmation",
        )

        with pytest.raises(DeletionConfirmationError):
            await _delete_memory_impl(
                engine,
                "usr_1",
                "mem_1",
                hard=True,
                conversation_id="cnv_1",
                platform_id="web",
            )
        with pytest.raises(DeletionConfirmationError):
            await _delete_conversation_impl(
                engine,
                "usr_1",
                "cnv_1",
                platform_id="web",
            )

        confirmation = await _delete_memory_impl(
            engine,
            "usr_1",
            "mem_1",
            hard=True,
            confirmation="HARD_DELETE_MEMORY",
            conversation_id="cnv_1",
            platform_id="web",
        )
        assert confirmation == "Hard-deleted memory mem_1."
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mcp_list_memories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        await engine.create_conversation(
            "usr_1",
            "cnv_1",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        await _seed_memory(
            engine,
            memory_id="mem_evidence",
            text="retry loop websocket backoff",
            object_type=MemoryObjectType.EVIDENCE,
        )
        await _seed_memory(
            engine,
            memory_id="mem_belief",
            text="The issue is likely in the retry guard.",
            object_type=MemoryObjectType.BELIEF,
        )
        await _seed_memory(
            engine,
            memory_id="mem_archived",
            text="Old archived memory",
            object_type=MemoryObjectType.EVIDENCE,
        )
        await _seed_memory(
            engine,
            memory_id="mem_pending",
            text="Pending memory",
            object_type=MemoryObjectType.EVIDENCE,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
        )
        await _seed_memory(
            engine,
            memory_id="mem_private",
            text="Private memory",
            object_type=MemoryObjectType.EVIDENCE,
            privacy_level=2,
        )
        await _seed_memory(
            engine,
            memory_id="mem_declined",
            text="Declined memory",
            object_type=MemoryObjectType.EVIDENCE,
            status=MemoryStatus.DECLINED,
        )
        await _delete_memory_impl(
            engine,
            "usr_1",
            "mem_archived",
            conversation_id="cnv_1",
            platform_id="web",
        )

        all_memories = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_1",
                platform_id="web",
            )
        )
        belief_memories = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                memory_type=MemoryObjectType.BELIEF.value,
                conversation_id="cnv_1",
                platform_id="web",
            )
        )

        assert len(all_memories) == 3
        assert {memory["id"] for memory in all_memories} == {
            "mem_evidence",
            "mem_belief",
            "mem_archived",
        }
        assert next(
            memory for memory in all_memories if memory["id"] == "mem_archived"
        )["status"] == (MemoryStatus.ARCHIVED.value)
        assert all(memory["id"] != "mem_pending" for memory in all_memories)
        assert all(memory["id"] != "mem_private" for memory in all_memories)
        assert all(memory["id"] != "mem_declined" for memory in all_memories)
        assert len(belief_memories) == 1
        assert belief_memories[0]["id"] == "mem_belief"
        assert belief_memories[0]["type"] == MemoryObjectType.BELIEF.value
    finally:
        await engine.close()


@pytest.mark.parametrize(
    ("space_id", "boundary_mode"),
    [
        ("space_vault", SpaceBoundaryMode.PRIVACY_VAULT),
        ("space_severed", SpaceBoundaryMode.SEVERANCE),
    ],
)
@pytest.mark.asyncio
async def test_mcp_memory_tools_enforce_space_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    space_id: str,
    boundary_mode: SpaceBoundaryMode,
) -> None:
    provider = MCPProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=tmp_path / "atagia-mcp.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )

    await engine.setup()
    try:
        await engine.create_user("usr_1")
        runtime = engine.runtime
        if runtime is None:
            raise AssertionError("Engine runtime should be initialized")
        connection = await runtime.open_connection()
        try:
            await SpaceRepository(connection, runtime.clock).resolve_space(
                owner_user_id="usr_1",
                space_id=space_id,
                boundary_mode=boundary_mode,
                display_name=space_id,
                source_kind="explicit",
                source_id=space_id,
            )
        finally:
            await connection.close()
        await engine.create_conversation(
            "usr_1",
            "cnv_outside",
            assistant_mode_id="coding_debug",
            platform_id="web",
        )
        await engine.create_conversation(
            "usr_1",
            "cnv_inside",
            assistant_mode_id="coding_debug",
            platform_id="web",
            space_id=space_id,
        )
        for operation in ("list", "edit", "delete"):
            await _seed_memory(
                engine,
                memory_id=f"mem_{space_id}_{operation}",
                text=f"{operation} mcp boundary token inside {space_id}",
                scope=MemoryScope.GLOBAL_USER,
                conversation_id=None,
                platform_id="web",
                space_id=space_id,
                space_boundary_mode=boundary_mode,
            )

        outside_list = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_outside",
                platform_id="web",
            )
        )
        outside_search = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "boundary",
                conversation_id="cnv_outside",
                platform_id="web",
            )
        )
        outside_ids = {memory["id"] for memory in outside_list}
        outside_search_ids = {memory["id"] for memory in outside_search}
        assert f"mem_{space_id}_list" not in outside_ids
        assert f"mem_{space_id}_list" not in outside_search_ids

        with pytest.raises(MemoryNotFoundError):
            await _edit_memory_impl(
                engine,
                "usr_1",
                f"mem_{space_id}_edit",
                "Outside MCP edit must not land.",
                conversation_id="cnv_outside",
                platform_id="web",
            )
        with pytest.raises(MemoryNotFoundError):
            await _delete_memory_impl(
                engine,
                "usr_1",
                f"mem_{space_id}_delete",
                conversation_id="cnv_outside",
                platform_id="web",
            )

        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            edit_memory = await memories.get_memory_object(
                f"mem_{space_id}_edit", "usr_1"
            )
            delete_memory = await memories.get_memory_object(
                f"mem_{space_id}_delete", "usr_1"
            )
            assert edit_memory is not None
            assert delete_memory is not None
            assert (
                edit_memory["canonical_text"]
                == f"edit mcp boundary token inside {space_id}"
            )
            assert delete_memory["status"] == MemoryStatus.ACTIVE.value
        finally:
            await connection.close()

        inside_list = json.loads(
            await _list_memories_impl(
                engine,
                "usr_1",
                conversation_id="cnv_inside",
                platform_id="web",
            )
        )
        inside_search = json.loads(
            await _search_memories_impl(
                engine,
                "usr_1",
                "boundary",
                conversation_id="cnv_inside",
                platform_id="web",
            )
        )
        inside_ids = {memory["id"] for memory in inside_list}
        inside_search_ids = {memory["id"] for memory in inside_search}
        assert f"mem_{space_id}_list" in inside_ids
        assert f"mem_{space_id}_list" in inside_search_ids

        edited_payload = json.loads(
            await _edit_memory_impl(
                engine,
                "usr_1",
                f"mem_{space_id}_edit",
                "Inside MCP edit is allowed.",
                conversation_id="cnv_inside",
                platform_id="web",
            )
        )
        assert edited_payload == {
            "id": f"mem_{space_id}_edit",
            "canonical_text": "Inside MCP edit is allowed.",
        }

        confirmation = await _delete_memory_impl(
            engine,
            "usr_1",
            f"mem_{space_id}_delete",
            conversation_id="cnv_inside",
            platform_id="web",
        )
        assert confirmation == f"Archived memory mem_{space_id}_delete."
        connection = await runtime.open_connection()
        try:
            memories = MemoryObjectRepository(connection, runtime.clock)
            archived_memory = await memories.get_memory_object(
                f"mem_{space_id}_delete", "usr_1"
            )
            assert archived_memory is not None
            assert archived_memory["status"] == MemoryStatus.ARCHIVED.value
        finally:
            await connection.close()
    finally:
        await engine.close()
