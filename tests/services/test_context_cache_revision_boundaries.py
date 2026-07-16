"""LIFE-02 cache-revision races across public mutation boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import json
from pathlib import Path
import threading
from typing import Any, Literal

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from atagia import Atagia
from atagia.app import AppRuntime, create_app
from atagia.core.config import Settings
from atagia.core.conversation_lifecycle_repository import (
    ConversationLifecycleIdentity,
    ConversationLifecycleRepository,
)
from atagia.core.memory_evidence_repository import MemoryEvidenceRepository
from atagia.core.memory_fact_facet_repository import MemoryFactFacetRepository
from atagia.core.redis_client import LIFECYCLE_MIRROR_PREFIX
from atagia.core.storage_backend import build_recent_window_key
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.user_lifecycle_repository import (
    UserLifecycleIdentity,
    UserLifecycleRepository,
)
from atagia.models.schemas_memory import (
    AdaptiveGateStatus,
    ComposedContext,
    MemoryEvidenceSupportKind,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    ResponseMode,
    RetrievalPlan,
)
from atagia.models.schemas_replay import PipelineResult
from atagia.services.context_cache_service import (
    AdaptiveContextResolution,
    ContextCacheService,
)
from atagia.services.lifecycle_mirror_reconciler import (
    reconcile_active_lifecycle_mirror,
)
from atagia.services.lifecycle_service import (
    ERASE_ALL_DATA_CONFIRMATION,
    HARD_DELETE_MEMORY_CONFIRMATION,
)
from atagia.services.llm_client import LLMClient
from atagia.services.retrieval_service import RetrievalService
from tests.redis_real_support import RealRedisServer, running_redis_server


MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)

USER_ID = "usr_life_02"
CONVERSATION_ID = "cnv_life_02"
MEMORY_ID = "mem_life_02"
OLD_TEXT = "OLD_LIFE_02_MARKER retry loop"
NEW_TEXT = "NEW_LIFE_02_MARKER retry loop"

MutationAction = Literal["edit", "delete"]
ReadMutationAction = Literal["edit", "delete", "erase"]
ReadPath = Literal["normal", "smart_fast"]


@pytest.fixture
def real_redis_server(tmp_path: Path) -> Iterator[RealRedisServer]:
    """Use the repository's isolated-server fixture and established skip."""

    with running_redis_server(tmp_path) as server:
        yield server


def _settings(tmp_path: Path, *, redis_url: str | None = None) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-life-02.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="redis" if redis_url is not None else "inprocess",
        redis_url=redis_url or "redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/reply-test-model",
        llm_ingest_model="openai/extract-test-model",
        llm_retrieval_model="openai/score-test-model",
        llm_component_models={"intent_classifier": "openai/classify-test-model"},
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        context_cache_enabled=True,
        debug=False,
        allow_insecure_http=True,
        small_corpus_token_threshold_ratio=0.0,
    )


async def _canonical_memory_retrieval(
    _service: RetrievalService,
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    conversation_id: str,
    message_text: str,
    mode: str | None = None,
    **_kwargs: Any,
) -> PipelineResult:
    """Return current canonical text so cache identity is the only race variable."""

    cursor = await connection.execute(
        """
        SELECT id, canonical_text
        FROM memory_objects
        WHERE id = ?
          AND user_id = ?
          AND status = 'active'
        """,
        (MEMORY_ID, user_id),
    )
    row = await cursor.fetchone()
    selected_memory_ids = [] if row is None else [str(row["id"])]
    memory_block = "" if row is None else str(row["canonical_text"])
    return PipelineResult(
        retrieval_plan=RetrievalPlan(
            original_query=message_text,
            assistant_mode_id=mode or "coding_debug",
            conversation_id=conversation_id,
            platform_id="web",
            max_candidates=1,
            max_context_items=1,
            privacy_ceiling=3,
        ),
        composed_context=ComposedContext(
            memory_block=memory_block,
            selected_memory_ids=selected_memory_ids,
            total_tokens_estimate=len(memory_block.split()),
            budget_tokens=128,
            items_included=len(selected_memory_ids),
            items_dropped=0,
        ),
        adaptive_gate_status=AdaptiveGateStatus.OFF_SHADOW,
    )


def _install_deterministic_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RetrievalService,
        "retrieve_with_connection",
        _canonical_memory_retrieval,
    )
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(),
    )


async def _seed(runtime: AppRuntime) -> None:
    connection = await runtime.open_connection()
    try:
        await UserRepository(connection, runtime.clock).create_user(USER_ID)
        await ConversationRepository(
            connection,
            runtime.clock,
        ).create_conversation(
            CONVERSATION_ID,
            USER_ID,
            None,
            "coding_debug",
            "LIFE-02 race",
            platform_id="web",
        )
        source = await MessageRepository(
            connection,
            runtime.clock,
        ).create_message(
            "msg_life_02_source",
            CONVERSATION_ID,
            "user",
            1,
            OLD_TEXT,
        )
        await MemoryObjectRepository(
            connection,
            runtime.clock,
        ).create_memory_object(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text=OLD_TEXT,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id=MEMORY_ID,
            platform_id="web",
            payload={"source_message_ids": [str(source["id"])]},
        )
        await MemoryEvidenceRepository(
            connection,
            runtime.clock,
        ).create_support_edge_with_spans(
            user_id=USER_ID,
            memory_id=MEMORY_ID,
            support_kind=MemoryEvidenceSupportKind.DIRECT,
            confidence=0.9,
            spans=[
                {
                    "span_role": "source",
                    "message_id": str(source["id"]),
                    "conversation_id": CONVERSATION_ID,
                    "quote_text": OLD_TEXT,
                }
            ],
        )
    finally:
        await connection.close()


async def _active_identity(runtime: AppRuntime) -> UserLifecycleIdentity:
    connection = await runtime.open_connection()
    try:
        identity = await UserLifecycleRepository(
            connection,
            runtime.clock,
        ).get_active_identity(USER_ID)
    finally:
        await connection.close()
    assert identity is not None
    return identity


async def _active_conversation_identity(
    runtime: AppRuntime,
) -> ConversationLifecycleIdentity:
    connection = await runtime.open_connection()
    try:
        identity = await ConversationLifecycleRepository(
            connection,
            runtime.clock,
        ).get_active_identity(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
        )
    finally:
        await connection.close()
    assert identity is not None
    return identity


async def _upsert_fact_facet(
    runtime: AppRuntime,
    *,
    value_text: str,
) -> dict[str, Any]:
    connection = await runtime.open_connection()
    try:
        cursor = await connection.execute(
            """
            SELECT id
            FROM memory_evidence_spans
            WHERE user_id = ?
              AND memory_id = ?
              AND message_id = ?
              AND span_role = 'source'
            """,
            (USER_ID, MEMORY_ID, "msg_life_02_source"),
        )
        source_span = await cursor.fetchone()
        assert source_span is not None
        return await MemoryFactFacetRepository(
            connection,
            runtime.clock,
        ).upsert_fact_facet(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            memory_id=MEMORY_ID,
            source_span_id=str(source_span["id"]),
            source_message_id="msg_life_02_source",
            subject_surface="user",
            surface_class="structured",
            facet_label="preference.retry_strategy",
            value_text=value_text,
            value_norm_key="retry strategy",
            assertion_kind="state",
            support_kind="direct",
            observed_at=runtime.clock.now().isoformat(),
            confidence=0.9,
        )
    finally:
        await connection.close()


async def _resolve(
    runtime: AppRuntime,
    *,
    response_mode: ResponseMode = ResponseMode.NORMAL,
    fast: bool = False,
) -> AdaptiveContextResolution:
    connection = await runtime.open_connection()
    try:
        service = ContextCacheService(runtime)
        if fast:
            return await service.resolve_fast_with_connection(
                connection,
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
                message_text="retry loop",
                response_mode=response_mode,
            )
        return await service.resolve_with_connection(
            connection,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            message_text="retry loop",
            response_mode=response_mode,
        )
    finally:
        await connection.close()


async def _capture_old_result(
    runtime: AppRuntime,
    *,
    response_mode: ResponseMode = ResponseMode.NORMAL,
) -> AdaptiveContextResolution:
    resolution = await _resolve(runtime, response_mode=response_mode)
    assert resolution.pending_cache_entry is not None
    assert resolution.cache_lifecycle_epoch is not None
    assert resolution.cache_lifecycle_cleanup_key is not None
    assert resolution.cache_revision is not None
    assert OLD_TEXT in resolution.pending_cache_entry.composed_context.memory_block
    return resolution


async def _leave_old_physical_entry(
    runtime: AppRuntime, resolution: AdaptiveContextResolution
) -> None:
    entry = resolution.pending_cache_entry
    assert entry is not None
    assert resolution.cache_ttl_seconds is not None
    await runtime.storage_backend.set_context_view(
        entry.cache_key,
        entry.model_dump(mode="json"),
        resolution.cache_ttl_seconds,
    )


def _make_user_cache_cleanup_fail(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AppRuntime,
) -> None:
    async def unavailable_cleanup(_backend: object, _user_id: str) -> int:
        raise ConnectionError("injected physical cache cleanup failure")

    monkeypatch.setattr(
        type(runtime.storage_backend),
        "delete_context_views_for_user",
        unavailable_cleanup,
    )


def _install_pre_identity_publish_barrier(
    monkeypatch: pytest.MonkeyPatch,
    old_resolution: AdaptiveContextResolution,
) -> tuple[threading.Event, threading.Event]:
    """Pause one old publish immediately before its SQLite identity check."""

    reached = threading.Event()
    resume = threading.Event()
    original = ContextCacheService._context_identity_is_current
    paused = False

    async def paused_identity_check(
        service: ContextCacheService,
        user_id: str,
        *,
        lifecycle_epoch: str,
        cache_revision: int,
        derivation_revision: int,
    ) -> bool:
        nonlocal paused
        if (
            not paused
            and user_id == USER_ID
            and lifecycle_epoch == old_resolution.cache_lifecycle_epoch
            and cache_revision == old_resolution.cache_revision
            and derivation_revision == old_resolution.source_derivation_revision
        ):
            paused = True
            reached.set()
            if not await asyncio.to_thread(resume.wait, 5.0):
                raise TimeoutError("stale cache publish barrier was not released")
        return await original(
            service,
            user_id,
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
        )

    monkeypatch.setattr(
        ContextCacheService,
        "_context_identity_is_current",
        paused_identity_check,
    )
    return reached, resume


def _install_post_fetch_read_barrier(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AppRuntime,
    *,
    cache_key: str,
) -> tuple[threading.Event, threading.Event]:
    """Pause after the backend returns old bytes but before SQLite revalidation."""

    reached = threading.Event()
    resume = threading.Event()
    backend_type = type(runtime.storage_backend)
    original = backend_type.get_context_view
    paused = False

    async def paused_get_context_view(
        backend: object,
        requested_cache_key: str,
    ) -> dict[str, Any] | None:
        nonlocal paused
        raw = await original(backend, requested_cache_key)
        if not paused and requested_cache_key == cache_key:
            paused = True
            assert raw is not None
            assert OLD_TEXT in json.dumps(raw, sort_keys=True)
            reached.set()
            if not await asyncio.to_thread(resume.wait, 5.0):
                raise TimeoutError("stale cache read barrier was not released")
        return raw

    monkeypatch.setattr(
        backend_type,
        "get_context_view",
        paused_get_context_view,
    )
    return reached, resume


async def _publish_old_result(
    runtime: AppRuntime,
    old_resolution: AdaptiveContextResolution,
) -> bool:
    return await ContextCacheService(runtime).publish_pending_cache_entry(
        old_resolution,
        last_retrieval_message_seq=1,
    )


async def _reconcile_current_redis_mirror(runtime: AppRuntime) -> UserLifecycleIdentity:
    identity = await _active_identity(runtime)
    connection = await runtime.open_connection()
    try:
        assert await reconcile_active_lifecycle_mirror(
            connection,
            runtime.storage_backend,
            user_id=USER_ID,
            lifecycle_epoch=identity.lifecycle_epoch,
            lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
        )
    finally:
        await connection.close()
    return identity


async def _assert_stale_publish_rejected(
    runtime: AppRuntime,
    old_resolution: AdaptiveContextResolution,
    action: MutationAction,
    old_physical_entry_remains: bool,
) -> None:
    current_identity = await _active_identity(runtime)
    assert current_identity.lifecycle_epoch == old_resolution.cache_lifecycle_epoch
    assert current_identity.lifecycle_cleanup_key == (
        old_resolution.cache_lifecycle_cleanup_key
    )
    assert old_resolution.cache_revision is not None
    assert current_identity.cache_revision > old_resolution.cache_revision

    service = ContextCacheService(runtime)
    old_raw = await runtime.storage_backend.get_context_view(
        str(old_resolution.cache_key)
    )
    if old_physical_entry_remains:
        assert old_raw is not None
        assert OLD_TEXT in json.dumps(old_raw, sort_keys=True)
    else:
        assert old_raw is None

    current_resolution = await _resolve(runtime)
    assert current_resolution.from_cache is False
    assert current_resolution.cache_key != old_resolution.cache_key
    current_block = current_resolution.composed_context.memory_block
    assert OLD_TEXT not in current_block
    if action == "edit":
        assert NEW_TEXT in current_block
        assert current_resolution.composed_context.selected_memory_ids == [MEMORY_ID]
    else:
        assert NEW_TEXT not in current_block
        assert MEMORY_ID not in current_resolution.composed_context.selected_memory_ids

    assert await service.publish_pending_cache_entry(
        current_resolution,
        last_retrieval_message_seq=2,
    )
    current_raw = await runtime.storage_backend.get_context_view(
        str(current_resolution.cache_key)
    )
    assert current_raw is not None
    serialized_current = json.dumps(current_raw, sort_keys=True)
    assert OLD_TEXT not in serialized_current
    if action == "edit":
        assert NEW_TEXT in serialized_current
    else:
        assert MEMORY_ID not in current_raw["selected_memory_ids"]


async def _mutate_through_library(engine: Atagia, action: MutationAction) -> None:
    if action == "edit":
        edited = await engine.edit_memory(USER_ID, MEMORY_ID, NEW_TEXT)
        assert edited["canonical_text"] == NEW_TEXT
        return
    report = await engine.delete_memory(
        USER_ID,
        MEMORY_ID,
        hard=True,
        confirmation=HARD_DELETE_MEMORY_CONFIRMATION,
    )
    assert report.deleted_memories == 1


async def _mutate_for_cache_read_race(
    engine: Atagia,
    action: ReadMutationAction,
) -> None:
    if action == "erase":
        report = await engine.erase_user_data(
            USER_ID,
            confirmation=ERASE_ALL_DATA_CONFIRMATION,
        )
        assert report.deleted_memories == 1
        assert report.deleted_conversations == 1
        return
    await _mutate_through_library(engine, action)


def _mutate_through_api(client: TestClient, action: MutationAction) -> None:
    common_payload = {
        "user_id": USER_ID,
        "conversation_id": CONVERSATION_ID,
        "platform_id": "web",
    }
    if action == "edit":
        response = client.patch(
            f"/v1/memories/{MEMORY_ID}",
            json={**common_payload, "canonical_text": NEW_TEXT},
        )
        assert response.status_code == 200, response.text
        assert response.json()["canonical_text"] == NEW_TEXT
        return
    response = client.post(
        f"/v1/memories/{MEMORY_ID}/delete",
        json={
            **common_payload,
            "hard": True,
            "confirmation": HARD_DELETE_MEMORY_CONFIRMATION,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["deleted_memories"] == 1


def _configure_library_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATAGIA_STORAGE_BACKEND", "inprocess")
    monkeypatch.setenv("ATAGIA_WORKERS_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_ENABLED", "true")
    monkeypatch.setenv("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0")


@pytest.mark.asyncio
@pytest.mark.parametrize("read_path", ["normal", "smart_fast"])
@pytest.mark.parametrize("action", ["edit", "delete", "erase"])
async def test_inflight_cache_read_rechecks_sqlite_after_backend_fetch(
    read_path: ReadPath,
    action: ReadMutationAction,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_retrieval(monkeypatch)
    _configure_library_environment(monkeypatch)
    engine = Atagia(
        db_path=tmp_path / f"cache-read-{read_path}-{action}.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        context_cache_enabled=True,
    )
    await engine.setup()
    try:
        runtime = engine.runtime
        assert runtime is not None
        await _seed(runtime)
        response_mode = (
            ResponseMode.NORMAL if read_path == "normal" else ResponseMode.SMART_FAST
        )
        old_resolution = await _capture_old_result(
            runtime,
            response_mode=response_mode,
        )
        assert await _publish_old_result(runtime, old_resolution)
        assert old_resolution.cache_key is not None
        reached, resume = _install_post_fetch_read_barrier(
            monkeypatch,
            runtime,
            cache_key=old_resolution.cache_key,
        )
        stale_read = asyncio.create_task(
            _resolve(
                runtime,
                response_mode=response_mode,
                fast=read_path == "smart_fast",
            )
        )

        assert await asyncio.to_thread(reached.wait, 5.0)
        try:
            await _mutate_for_cache_read_race(engine, action)
        finally:
            resume.set()
        resolution = await stale_read

        assert OLD_TEXT not in resolution.composed_context.memory_block
        if read_path == "normal":
            assert resolution.from_cache is False
            if action == "edit":
                assert NEW_TEXT in resolution.composed_context.memory_block
            else:
                assert MEMORY_ID not in resolution.composed_context.selected_memory_ids
        else:
            assert (
                resolution.source_retrieval_plan["smart_fast_warm_entry_present"]
                is False
            )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_fact_facet_mutations_reject_pending_publish_and_cached_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_retrieval(monkeypatch)
    _configure_library_environment(monkeypatch)
    engine = Atagia(
        db_path=tmp_path / "fact-facet-cache-revision.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        context_cache_enabled=True,
    )
    await engine.setup()
    try:
        runtime = engine.runtime
        assert runtime is not None
        await _seed(runtime)

        pending_before_insert = await _capture_old_result(runtime)
        identity_before_insert = await _active_identity(runtime)
        await _upsert_fact_facet(runtime, value_text="Prefer bounded retries")
        identity_after_insert = await _active_identity(runtime)
        assert identity_after_insert.cache_revision == (
            identity_before_insert.cache_revision + 1
        )
        assert identity_after_insert.source_revision == (
            identity_before_insert.source_revision
        )
        assert not await _publish_old_result(runtime, pending_before_insert)
        assert (
            await runtime.storage_backend.get_context_view(
                str(pending_before_insert.cache_key)
            )
            is None
        )

        current = await _resolve(runtime)
        assert current.from_cache is False
        assert await _publish_old_result(runtime, current)
        stale_physical_entry = await runtime.storage_backend.get_context_view(
            str(current.cache_key)
        )
        assert stale_physical_entry is not None

        identity_before_update = await _active_identity(runtime)
        await _upsert_fact_facet(runtime, value_text="Prefer one bounded retry")
        identity_after_update = await _active_identity(runtime)
        assert identity_after_update.cache_revision == (
            identity_before_update.cache_revision + 1
        )
        assert identity_after_update.source_revision == (
            identity_before_update.source_revision
        )

        refreshed = await _resolve(runtime)
        assert refreshed.from_cache is False
        assert refreshed.cache_key != current.cache_key
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["edit", "delete"])
async def test_library_inprocess_rejects_stale_publish_with_cleanup_failure(
    action: MutationAction,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_retrieval(monkeypatch)
    _configure_library_environment(monkeypatch)
    engine = Atagia(
        db_path=tmp_path / f"library-inprocess-{action}.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        context_cache_enabled=True,
    )
    await engine.setup()
    try:
        runtime = engine.runtime
        assert runtime is not None
        await _seed(runtime)
        old_resolution = await _capture_old_result(runtime)
        await _leave_old_physical_entry(runtime, old_resolution)
        _make_user_cache_cleanup_fail(monkeypatch, runtime)
        reached, resume = _install_pre_identity_publish_barrier(
            monkeypatch,
            old_resolution,
        )
        stale_publish = asyncio.create_task(
            _publish_old_result(runtime, old_resolution)
        )

        assert await asyncio.to_thread(reached.wait, 5.0)
        try:
            await _mutate_through_library(engine, action)
        finally:
            resume.set()
        assert not await stale_publish

        await _assert_stale_publish_rejected(
            runtime,
            old_resolution=old_resolution,
            action=action,
            old_physical_entry_remains=True,
        )
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["edit", "delete"])
async def test_library_redis_reset_rejects_stale_publish_from_sqlite_revision(
    action: MutationAction,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_redis_server: RealRedisServer,
) -> None:
    _install_deterministic_retrieval(monkeypatch)
    _configure_library_environment(monkeypatch)
    engine = Atagia(
        db_path=tmp_path / f"library-redis-{action}.db",
        redis_url=real_redis_server.url,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        context_cache_enabled=True,
    )
    await engine.setup()
    try:
        runtime = engine.runtime
        assert runtime is not None
        await _seed(runtime)
        old_resolution = await _capture_old_result(runtime)
        reached, resume = _install_pre_identity_publish_barrier(
            monkeypatch,
            old_resolution,
        )
        stale_publish = asyncio.create_task(
            _publish_old_result(runtime, old_resolution)
        )

        assert await asyncio.to_thread(reached.wait, 5.0)
        try:
            await _mutate_through_library(engine, action)
            real_redis_server.client.flushall()
            current_identity = await _reconcile_current_redis_mirror(runtime)
            assert (
                real_redis_server.client.get(
                    f"{LIFECYCLE_MIRROR_PREFIX}{current_identity.lifecycle_cleanup_key}"
                )
                == f"active:{current_identity.lifecycle_epoch}"
            )
        finally:
            resume.set()
        assert not await stale_publish
        await _assert_stale_publish_rejected(
            runtime,
            old_resolution=old_resolution,
            action=action,
            old_physical_entry_remains=False,
        )
    finally:
        await engine.close()


@pytest.mark.parametrize("action", ["edit", "delete"])
def test_api_inprocess_rejects_stale_publish_with_cleanup_failure(
    action: MutationAction,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_retrieval(monkeypatch)
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        client.portal.call(_seed, runtime)
        old_resolution = client.portal.call(_capture_old_result, runtime)
        client.portal.call(_leave_old_physical_entry, runtime, old_resolution)
        _make_user_cache_cleanup_fail(monkeypatch, runtime)
        reached, resume = _install_pre_identity_publish_barrier(
            monkeypatch,
            old_resolution,
        )
        stale_publish = client.portal.start_task_soon(
            _publish_old_result,
            runtime,
            old_resolution,
        )

        assert reached.wait(5.0)
        try:
            _mutate_through_api(client, action)
        finally:
            resume.set()
        assert not stale_publish.result(timeout=5.0)

        client.portal.call(
            _assert_stale_publish_rejected,
            runtime,
            old_resolution,
            action,
            True,
        )


@pytest.mark.parametrize("action", ["edit", "delete"])
def test_api_redis_reset_rejects_stale_publish_from_sqlite_revision(
    action: MutationAction,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_redis_server: RealRedisServer,
) -> None:
    _install_deterministic_retrieval(monkeypatch)
    app = create_app(_settings(tmp_path, redis_url=real_redis_server.url))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        client.portal.call(_seed, runtime)
        old_resolution = client.portal.call(_capture_old_result, runtime)
        reached, resume = _install_pre_identity_publish_barrier(
            monkeypatch,
            old_resolution,
        )
        stale_publish = client.portal.start_task_soon(
            _publish_old_result,
            runtime,
            old_resolution,
        )

        assert reached.wait(5.0)
        try:
            _mutate_through_api(client, action)
            real_redis_server.client.flushall()
            current_identity = client.portal.call(
                _reconcile_current_redis_mirror,
                runtime,
            )
            assert (
                real_redis_server.client.get(
                    f"{LIFECYCLE_MIRROR_PREFIX}{current_identity.lifecycle_cleanup_key}"
                )
                == f"active:{current_identity.lifecycle_epoch}"
            )
        finally:
            resume.set()
        assert not stale_publish.result(timeout=5.0)
        client.portal.call(
            _assert_stale_publish_rejected,
            runtime,
            old_resolution,
            action,
            False,
        )


@pytest.mark.asyncio
async def test_recent_window_publish_removes_stale_write_after_revision_bump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_retrieval(monkeypatch)
    _configure_library_environment(monkeypatch)
    engine = Atagia(
        db_path=tmp_path / "recent-window-revision-race.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        context_cache_enabled=True,
    )
    await engine.setup()
    try:
        runtime = engine.runtime
        assert runtime is not None
        await _seed(runtime)
        identity = await _active_identity(runtime)
        conversation_identity = await _active_conversation_identity(runtime)
        reached = asyncio.Event()
        resume = asyncio.Event()
        backend_type = type(runtime.storage_backend)
        original_publish = backend_type.set_recent_window_for_lifecycle

        async def publish_after_bump(
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
            reached.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            return await original_publish(
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

        monkeypatch.setattr(
            backend_type,
            "set_recent_window_for_lifecycle",
            publish_after_bump,
        )
        stale_publish = asyncio.create_task(
            ContextCacheService(runtime).publish_recent_window(
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
                messages=[{"role": "user", "content": OLD_TEXT}],
                lifecycle_epoch=identity.lifecycle_epoch,
                lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
                cache_revision=identity.cache_revision,
                derivation_revision=identity.derivation_revision,
                conversation_lifecycle_epoch=(conversation_identity.lifecycle_epoch),
                conversation_source_revision=conversation_identity.source_revision,
            )
        )

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        connection = await runtime.open_connection()
        try:
            bumped = await UserLifecycleRepository(
                connection,
                runtime.clock,
            ).bump_cache_revision(
                USER_ID,
                expected_lifecycle_epoch=identity.lifecycle_epoch,
            )
            assert bumped == identity.cache_revision + 1
        finally:
            await connection.close()
        await runtime.storage_backend.delete_recent_window_for_conversation(
            USER_ID,
            CONVERSATION_ID,
        )
        resume.set()

        assert not await stale_publish
        assert (
            await runtime.storage_backend.get_recent_window(
                build_recent_window_key(USER_ID, CONVERSATION_ID)
            )
            is None
        )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_stale_recent_window_cleanup_cannot_delete_newer_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_retrieval(monkeypatch)
    _configure_library_environment(monkeypatch)
    engine = Atagia(
        db_path=tmp_path / "recent-window-conditional-cleanup-race.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        context_cache_enabled=True,
    )
    await engine.setup()
    try:
        runtime = engine.runtime
        assert runtime is not None
        await _seed(runtime)
        old_identity = await _active_identity(runtime)
        conversation_identity = await _active_conversation_identity(runtime)
        old_write_completed = asyncio.Event()
        release_old_publisher = asyncio.Event()
        backend_type = type(runtime.storage_backend)
        original_publish = backend_type.set_recent_window_for_lifecycle

        async def pause_old_after_write(
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
            if messages == [{"role": "user", "content": OLD_TEXT}]:
                old_write_completed.set()
                await asyncio.wait_for(release_old_publisher.wait(), timeout=5.0)
            return published

        monkeypatch.setattr(
            backend_type,
            "set_recent_window_for_lifecycle",
            pause_old_after_write,
        )
        old_publish = asyncio.create_task(
            ContextCacheService(runtime).publish_recent_window(
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
                messages=[{"role": "user", "content": OLD_TEXT}],
                lifecycle_epoch=old_identity.lifecycle_epoch,
                lifecycle_cleanup_key=old_identity.lifecycle_cleanup_key,
                cache_revision=old_identity.cache_revision,
                derivation_revision=old_identity.derivation_revision,
                conversation_lifecycle_epoch=(conversation_identity.lifecycle_epoch),
                conversation_source_revision=conversation_identity.source_revision,
            )
        )
        await asyncio.wait_for(old_write_completed.wait(), timeout=5.0)

        connection = await runtime.open_connection()
        try:
            next_revision = await UserLifecycleRepository(
                connection,
                runtime.clock,
            ).bump_cache_revision(
                USER_ID,
                expected_lifecycle_epoch=old_identity.lifecycle_epoch,
            )
            assert next_revision == old_identity.cache_revision + 1
        finally:
            await connection.close()
        new_identity = await _active_identity(runtime)
        assert new_identity.cache_revision == next_revision

        assert await ContextCacheService(runtime).publish_recent_window(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            messages=[{"role": "assistant", "content": NEW_TEXT}],
            lifecycle_epoch=new_identity.lifecycle_epoch,
            lifecycle_cleanup_key=new_identity.lifecycle_cleanup_key,
            cache_revision=new_identity.cache_revision,
            derivation_revision=new_identity.derivation_revision,
            conversation_lifecycle_epoch=conversation_identity.lifecycle_epoch,
            conversation_source_revision=conversation_identity.source_revision,
        )
        release_old_publisher.set()
        assert not await old_publish
        assert await runtime.storage_backend.get_recent_window(
            build_recent_window_key(USER_ID, CONVERSATION_ID)
        ) == [{"role": "assistant", "content": NEW_TEXT}]
    finally:
        await engine.close()
