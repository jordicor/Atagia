"""Tests for prepared initial-context package refresh work."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from atagia.core import json_utils
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import close_connection, initialize_database, open_connection
from atagia.core.initial_context_package_repository import (
    InitialContextPackageRepository,
    InitialContextPackageSourceChangedError,
)
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_initial_context_package import InitialContextPackageKind
from atagia.models.schemas_jobs import (
    InitialContextPackageRefreshReason,
)
from atagia.models.schemas_memory import (
    OperationalProfileSnapshot,
    OperationalRiskLevel,
    OperationalSignals,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
)
from atagia.services.initial_context_package_refresh_service import (
    InitialContextPackageRefreshEnqueuer,
)
from atagia.services.job_execution_context import StaleParentJobFenceError
from atagia.services.lifecycle_service import ConversationLifecycleService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.workers.initial_context_package_worker import InitialContextPackageWorker
from tests.durable_job_support import DurableJobTestBackend, bound_test_job_claim

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=True,
        debug=False,
        allow_insecure_http=True,
    )


async def _seed_runtime(
    database_path: str = ":memory:",
) -> tuple[aiosqlite.Connection, FrozenClock, ManifestLoader]:
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 6, 8, 9, 0, tzinfo=timezone.utc))
    manifest_loader = ManifestLoader(MANIFESTS_DIR)
    await sync_assistant_modes(connection, manifest_loader.load_all(), clock)
    await UserRepository(connection, clock).create_user("usr_1")
    await WorkspaceRepository(connection, clock).create_workspace(
        "wrk_1",
        "usr_1",
        "Workspace",
    )
    await ConversationRepository(connection, clock).create_conversation(
        "cnv_1",
        "usr_1",
        "wrk_1",
        "coding_debug",
        "Active chat",
        user_persona_id="persona_jordi",
        platform_id="aurvek",
        character_id="assistant_alpha",
    )
    return connection, clock, manifest_loader


async def _active_package_count(
    connection: aiosqlite.Connection,
    *,
    package_kind: InitialContextPackageKind,
) -> int:
    cursor = await connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM initial_context_packages
        WHERE user_id = ?
          AND package_kind = ?
          AND build_status = 'active'
        """,
        ("usr_1", package_kind.value),
    )
    row = await cursor.fetchone()
    return int(row["count"])


async def _package_status_counts(
    connection: aiosqlite.Connection,
) -> dict[str, int]:
    cursor = await connection.execute(
        """
        SELECT build_status, COUNT(*) AS count
        FROM initial_context_packages
        WHERE user_id = ?
        GROUP BY build_status
        """,
        ("usr_1",),
    )
    return {
        str(row["build_status"]): int(row["count"]) for row in await cursor.fetchall()
    }


def _operational_snapshot(token: str) -> OperationalProfileSnapshot:
    return OperationalProfileSnapshot(
        profile_id="default",
        signals=OperationalSignals(),
        risk_level=OperationalRiskLevel.NORMAL,
        authorized=True,
        profile_hash=f"profile-{token}",
        token=token,
    )


class CurationProvider(LLMProvider):
    name = "initial-context-worker-curation-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        assert request.metadata["purpose"] == "initial_context_package_curation"
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json_utils.dumps(
                {
                    "items": [
                        {
                            "candidate_ids": ["memory:mem_worker"],
                            "text": "Worker refresh preserved a curated package orientation.",
                            "status": "current",
                            "salience": 0.8,
                        }
                    ],
                    "nothing_to_add": False,
                },
                sort_keys=True,
            ),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in this test: {request.model}")


class FailingPackageBuilder:
    def __init__(self, method_name: str) -> None:
        self.method_name = method_name

    async def build_baseline_package(self, **_: object) -> object:
        if self.method_name == "build_baseline_package":
            raise RuntimeError("forced baseline build failure")
        raise AssertionError("Baseline build was not expected")

    async def build_conversation_package(self, **_: object) -> object:
        if self.method_name == "build_conversation_package":
            raise RuntimeError("forced conversation build failure")
        raise AssertionError("Conversation build was not expected")


class SourceChangedPackageBuilder:
    async def build_baseline_package(self, **_: object) -> object:
        raise InitialContextPackageSourceChangedError("source changed at activation")

    async def build_conversation_package(self, **_: object) -> object:
        raise InitialContextPackageSourceChangedError("source changed at activation")


async def _attach_transcript_rebuild(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
    *,
    job_id: str,
) -> str:
    workflow_id = "trw_icp_lock_retry"
    now = clock.now().isoformat()
    await connection.execute(
        """
        INSERT INTO transcript_rebuild_workflows(
            id, operation_id, user_id, conversation_id, selection_epoch,
            transcript_hash, mutation_kind, selected_message_ids_json,
            abandoned_message_ids_json, supporting_message_ids_json,
            affected_memory_ids_json, affected_summary_ids_json,
            orchestrator_job_id, stage, start_derivation_revision,
            created_at, updated_at
        ) VALUES (?, 'op_icp_lock_retry', 'usr_1', 'cnv_1', 1,
                  'hash_icp_lock_retry', 'replace', '[]', '[]', '[]', '[]',
                  '[]', 'orchestrator_icp_lock_retry', 'sources', 0, ?, ?)
        """,
        (workflow_id, now, now),
    )
    await connection.execute(
        """
        UPDATE worker_job_runs
        SET transcript_rebuild_id = ?,
            recovery_envelope_json = json_set(
                recovery_envelope_json,
                '$.transcript_rebuild_id',
                ?
            )
        WHERE job_id = ?
        """,
        (workflow_id, workflow_id, job_id),
    )
    await connection.commit()
    return workflow_id


@pytest.mark.asyncio
async def test_icp_process_job_rejects_unclaimed_and_mismatched_invocations() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        job_id = await InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        ).enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
            dispatch=False,
        )
        assert job_id is not None
        job = await backend.job_tracking_service.get_job_run(job_id)
        assert job is not None
        target = job["recovery_envelope_json"]
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )

        with pytest.raises(StaleParentJobFenceError):
            await worker.process_job(target)

        other = json_utils.loads(json_utils.dumps(target, sort_keys=True))
        other["job_id"] = "job_other_icp_claim"
        async with bound_test_job_claim(
            connection,
            backend,
            clock,
            other,
        ):
            with pytest.raises(StaleParentJobFenceError):
                await worker.process_job(target)

        assert backend._lifecycle_locks == {}
        assert await _package_status_counts(connection) == {}
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_icp_worker_defers_when_lifecycle_mirror_disappears_after_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        job_id = await InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        ).enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        assert job_id is not None
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        original_claim = worker._job_tracking.claim_notification

        async def claim_then_drop_mirror(
            stream_message: object,
            *,
            owner_id: str,
        ):
            claim = await original_claim(stream_message, owner_id=owner_id)  # type: ignore[arg-type]
            if claim is not None:
                backend._lifecycle_mirrors.pop(claim.lifecycle_cleanup_key, None)
            return claim

        monkeypatch.setattr(
            worker._job_tracking,
            "claim_notification",
            claim_then_drop_mirror,
        )

        result = await worker.run_once()
        stored = await backend.job_tracking_service.get_job_run(job_id)

        assert result.deferred == 1
        assert result.failed == 0
        assert stored is not None
        assert stored["status"] == "deferred"
        assert stored["recovery_envelope_json"] is not None
        assert await _package_status_counts(connection) == {}
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_selected_transcript_icp_lock_contention_defers_until_package_is_current() -> (
    None
):
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        job_id = await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.SOURCE_CHANGED,
            dispatch=False,
        )
        assert job_id is not None
        job_run = await backend.job_tracking_service.get_job_run(job_id)
        assert job_run is not None
        await _attach_transcript_rebuild(connection, clock, job_id=job_id)
        await backend.dispatch_durable_jobs()

        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        lock_key = worker._lock_key(
            user_id="usr_1",
            conversation_id="cnv_1",
            package_kind="all",
            retrieval_profile_id="coding_debug",
            privacy_enforcement="enforce",
            operational_profile_token=None,
        )
        lock_token = await backend.acquire_lock(
            lock_key,
            ttl_seconds=60,
            lifecycle_cleanup_key=str(job_run["lifecycle_cleanup_key"]),
            lifecycle_epoch=str(job_run["lifecycle_epoch"]),
        )
        assert lock_token is not None

        contended = await worker.run_once()
        assert contended.failed == 0
        assert contended.deferred == 1
        retrying = await backend.job_tracking_service.get_job_run(job_id)
        assert retrying is not None
        assert retrying["status"] == "deferred"
        assert retrying["recovery_envelope_json"] is not None
        assert (retrying["metadata_json"] or {}).get("status") is None

        await backend.release_lock(
            lock_key,
            lock_token,
            lifecycle_cleanup_key=str(job_run["lifecycle_cleanup_key"]),
            lifecycle_epoch=str(job_run["lifecycle_epoch"]),
        )
        await backend.advance_to_next_retry()
        completed = await worker.run_once()
        assert completed.acked == 1
        assert completed.failed == 0

        terminal = await backend.job_tracking_service.get_job_run(job_id)
        assert terminal is not None
        assert terminal["status"] == "succeeded"
        assert terminal["metadata_json"]["status"] == "refreshed"
        assert terminal["metadata_json"]["reason"] == "source_changed"
        cursor = await connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM initial_context_packages AS package
            JOIN user_lifecycles AS user_lifecycle
              ON user_lifecycle.user_id = package.user_id
            JOIN conversation_lifecycles AS conversation_lifecycle
              ON conversation_lifecycle.user_id = package.user_id
             AND conversation_lifecycle.conversation_id = package.conversation_id
            WHERE package.user_id = 'usr_1'
              AND package.conversation_id = 'cnv_1'
              AND package.package_kind = 'conversation'
              AND package.build_status = 'active'
              AND package.source_user_revision = user_lifecycle.source_revision
              AND package.source_conversation_revision =
                  conversation_lifecycle.source_revision
            """
        )
        assert int((await cursor.fetchone())["count"]) == 1
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_source_changed_successor_is_inserted_after_lock_and_dedupe_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        old_job_id = await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.SOURCE_CHANGED,
        )
        assert old_job_id is not None
        old_before = await backend.job_tracking_service.get_job_run(old_job_id)
        assert old_before is not None
        old_envelope = old_before["recovery_envelope_json"]
        old_generation = int(old_envelope["payload"]["refresh_generation"])

        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        worker._builder = SourceChangedPackageBuilder()  # type: ignore[assignment]
        original_enqueue = worker._enqueue_source_successor
        lock_key = worker._lock_key(
            user_id="usr_1",
            conversation_id="cnv_1",
            package_kind="all",
            retrieval_profile_id="coding_debug",
            privacy_enforcement="enforce",
            operational_profile_token=None,
        )

        async def assert_released_then_replace(**kwargs: object) -> str | None:
            probe_token = await backend.acquire_lock(
                lock_key,
                ttl_seconds=5,
                lifecycle_cleanup_key=str(old_before["lifecycle_cleanup_key"]),
                lifecycle_epoch=str(old_before["lifecycle_epoch"]),
            )
            assert probe_token is not None
            await backend.release_lock(
                lock_key,
                probe_token,
                lifecycle_cleanup_key=str(old_before["lifecycle_cleanup_key"]),
                lifecycle_epoch=str(old_before["lifecycle_epoch"]),
            )
            running = await backend.job_tracking_service.get_job_run(old_job_id)
            assert running is not None
            assert running["status"] == "running"
            successor_job_id = await original_enqueue(**kwargs)  # type: ignore[arg-type]
            terminal = await backend.job_tracking_service.get_job_run(old_job_id)
            assert terminal is not None
            assert terminal["status"] == "succeeded"
            return successor_job_id

        monkeypatch.setattr(
            worker,
            "_enqueue_source_successor",
            assert_released_then_replace,
        )
        iteration = await worker.run_once()
        assert iteration.acked == 1
        assert iteration.failed == 0

        rows = await (
            await connection.execute(
                """
                SELECT job_id, status, recovery_envelope_json
                FROM worker_job_runs
                WHERE user_id = ?
                  AND job_type = 'refresh_initial_context_package'
                ORDER BY _rowid ASC
                """,
                ("usr_1",),
            )
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["job_id"] == old_job_id
        assert rows[0]["status"] == "succeeded"
        assert rows[0]["recovery_envelope_json"] is None
        assert rows[1]["status"] in {"queued", "awaiting_claim"}
        successor_envelope = json_utils.loads(rows[1]["recovery_envelope_json"])
        successor_payload = successor_envelope["payload"]
        assert successor_payload["reason"] == "source_changed"
        assert int(successor_payload["refresh_generation"]) > old_generation
        assert (
            successor_payload["refresh_dedupe_key"]
            != (old_envelope["payload"]["refresh_dedupe_key"])
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_restart_between_source_cas_failure_and_successor_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "icp-source-successor-restart.db")
    connection, clock, manifest_loader = await _seed_runtime(database_path)
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        old_job_id = await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        assert old_job_id is not None
        crashing_worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        crashing_worker._builder = SourceChangedPackageBuilder()  # type: ignore[assignment]

        async def crash_before_successor(**_: object) -> str | None:
            raise RuntimeError("simulated restart before successor transaction")

        monkeypatch.setattr(
            crashing_worker,
            "_enqueue_source_successor",
            crash_before_successor,
        )
        first_iteration = await crashing_worker.run_once()
        assert first_iteration.failed == 1
        retrying = await backend.job_tracking_service.get_job_run(old_job_id)
        assert retrying is not None
        assert retrying["status"] == "retrying"
        assert retrying["recovery_envelope_json"] is not None
    finally:
        await backend.close()
        await connection.close()

    reopened = await initialize_database(database_path, MIGRATIONS_DIR)
    restarted_backend = DurableJobTestBackend(
        reopened,
        clock,
        settings=_settings(),
    )
    try:
        persisted_retry = await restarted_backend.job_tracking_service.get_job_run(
            old_job_id
        )
        assert persisted_retry is not None
        assert persisted_retry["status"] == "retrying"
        assert persisted_retry["recovery_envelope_json"] is not None

        await restarted_backend.advance_to_next_retry()
        restarted_worker = InitialContextPackageWorker(
            restarted_backend,
            reopened,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        restarted_worker._builder = SourceChangedPackageBuilder()  # type: ignore[assignment]
        second_iteration = await restarted_worker.run_once()
        assert second_iteration.acked == 1
        terminal = await restarted_backend.job_tracking_service.get_job_run(old_job_id)
        assert terminal is not None
        assert terminal["status"] == "succeeded"

        fresh_worker = InitialContextPackageWorker(
            restarted_backend,
            reopened,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        third_iteration = await fresh_worker.run_once()
        assert third_iteration.acked == 1
        assert (
            await _active_package_count(
                reopened,
                package_kind=InitialContextPackageKind.BASELINE,
            )
            == 1
        )
        assert (
            await _active_package_count(
                reopened,
                package_kind=InitialContextPackageKind.CONVERSATION,
            )
            == 1
        )
    finally:
        await restarted_backend.close()
        await reopened.close()


@pytest.mark.asyncio
async def test_refresh_enqueuer_coalesces_and_worker_materializes_packages() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        await MessageRepository(connection, clock).create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "Estamos preparando el paquete inicial.",
        )
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        first_job_id = await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            source_message_ids=["msg_1"],
        )
        second_job_id = await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            source_message_ids=["msg_1"],
        )

        assert first_job_id is not None
        assert second_job_id is None

        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        result = await worker.run_once()

        assert result.acked == 1
        repository = InitialContextPackageRepository(connection, clock)
        conversation_package = await repository.get_latest_for_conversation(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
        )
        assert conversation_package is not None
        assert (
            conversation_package.blocks_json.recent_verbatim_seed[0]["message_id"]
            == "msg_1"
        )
        assert (
            await _active_package_count(
                connection,
                package_kind=InitialContextPackageKind.BASELINE,
            )
            == 1
        )
        assert (
            await _active_package_count(
                connection,
                package_kind=InitialContextPackageKind.CONVERSATION,
            )
            == 1
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_direct_refresh_captures_coordinates_before_canonical_conversation_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "icp-worker-direct-capture.db")
    connection, clock, manifest_loader = await _seed_runtime(database_path)
    writer = await open_connection(database_path)
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    resume_refresh = asyncio.Event()
    try:
        await InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        ).enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        original_capture = worker._capture_source_coordinates  # noqa: SLF001
        coordinates_captured = asyncio.Event()

        async def hold_after_capture(**kwargs: object):
            coordinates = await original_capture(**kwargs)  # type: ignore[arg-type]
            coordinates_captured.set()
            await resume_refresh.wait()
            return coordinates

        monkeypatch.setattr(worker, "_capture_source_coordinates", hold_after_capture)
        iteration_task = asyncio.create_task(worker.run_once())
        await asyncio.wait_for(coordinates_captured.wait(), timeout=2)
        await ConversationRepository(writer, clock).set_conversation_incognito(
            "cnv_1",
            "usr_1",
            True,
        )
        resume_refresh.set()
        iteration = await iteration_task

        assert iteration.acked == 1
        row = await (
            await connection.execute(
                """
                SELECT coordinate_signature_json
                FROM initial_context_packages
                WHERE user_id = ?
                  AND package_kind = 'baseline'
                  AND build_status = 'active'
                ORDER BY refresh_generation DESC
                LIMIT 1
                """,
                ("usr_1",),
            )
        ).fetchone()
        assert row is not None
        signature = json_utils.loads(row["coordinate_signature_json"])
        assert signature["markers_json"]["lifecycle"]["incognito"] is True
    finally:
        resume_refresh.set()
        await backend.close()
        await close_connection(writer)
        await close_connection(connection)


@pytest.mark.asyncio
async def test_bulk_refresh_refetches_each_discovered_conversation_after_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = str(tmp_path / "icp-worker-bulk-refetch.db")
    connection, clock, manifest_loader = await _seed_runtime(database_path)
    writer = await open_connection(database_path)
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    resume_discovery = asyncio.Event()
    try:
        await InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        ).enqueue_refresh(
            user_id="usr_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        original_list = worker._conversation_repository.list_conversations  # noqa: SLF001
        stale_discovery_ready = asyncio.Event()

        async def hold_stale_discovery(*args: object, **kwargs: object):
            rows = await original_list(*args, **kwargs)  # type: ignore[arg-type]
            stale_discovery_ready.set()
            await resume_discovery.wait()
            return rows

        monkeypatch.setattr(
            worker._conversation_repository,  # noqa: SLF001
            "list_conversations",
            hold_stale_discovery,
        )
        iteration_task = asyncio.create_task(worker.run_once())
        await asyncio.wait_for(stale_discovery_ready.wait(), timeout=2)
        await ConversationRepository(writer, clock).set_conversation_incognito(
            "cnv_1",
            "usr_1",
            True,
        )
        resume_discovery.set()
        iteration = await iteration_task

        assert iteration.acked == 1
        rows = await (
            await connection.execute(
                """
                SELECT package_kind, coordinate_signature_json
                FROM initial_context_packages
                WHERE user_id = ?
                  AND build_status = 'active'
                ORDER BY package_kind
                """,
                ("usr_1",),
            )
        ).fetchall()
        assert {str(row["package_kind"]) for row in rows} == {
            "baseline",
            "conversation",
        }
        assert all(
            json_utils.loads(row["coordinate_signature_json"])["markers_json"][
                "lifecycle"
            ]["incognito"]
            is True
            for row in rows
        )
    finally:
        resume_discovery.set()
        await backend.close()
        await close_connection(writer)
        await close_connection(connection)


@pytest.mark.asyncio
async def test_worker_uses_llm_curation_for_background_package_refresh() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    provider = CurationProvider()
    try:
        await MessageRepository(connection, clock).create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "Estamos preparando el paquete inicial.",
        )
        await MemoryObjectRepository(connection, clock).create_memory_object(
            user_id="usr_1",
            memory_id="mem_worker",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.USER,
            canonical_text="Worker refresh has a source fact for curation.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            stability=0.8,
            vitality=0.8,
            maya_score=0.2,
            privacy_level=0,
            assistant_mode_id="coding_debug",
            workspace_id="wrk_1",
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            scope_canonical=MemoryScope.USER.value,
        )
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            source_message_ids=["msg_1"],
        )
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=replace(
                _settings(),
                initial_context_package_curation_enabled=True,
                llm_component_models={
                    "initial_context_package_curation": "openai/curation-test-model",
                },
            ),
            llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        )

        result = await worker.run_once()

        assert result.acked == 1
        assert provider.requests
        repository = InitialContextPackageRepository(connection, clock)
        conversation_package = await repository.get_latest_for_conversation(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
        )
        assert conversation_package is not None
        assert conversation_package.blocks_json.curated_items
        assert "curated package orientation" in (
            conversation_package.blocks_json.curated_orientation_block
        )
        assert (
            conversation_package.source_refs_json["curated_orientation"][0]["memory_id"]
            == "mem_worker"
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_refresh_variants_do_not_coalesce_or_stale_each_other() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        await MessageRepository(connection, clock).create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "Estamos preparando variantes del paquete inicial.",
        )
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        off_job_id = await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            source_message_ids=["msg_1"],
            privacy_enforcement="off",
            operational_profile=_operational_snapshot("offline"),
        )
        enforce_job_id = await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            source_message_ids=["msg_1"],
            privacy_enforcement="enforce",
            operational_profile=_operational_snapshot("offline"),
        )
        second_profile_job_id = await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            source_message_ids=["msg_1"],
            privacy_enforcement="off",
            operational_profile=_operational_snapshot("online"),
        )

        assert off_job_id is not None
        assert enforce_job_id is not None
        assert second_profile_job_id is not None

        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        assert (await worker.run_once()).acked == 1
        assert (await worker.run_once()).acked == 1
        assert (await worker.run_once()).acked == 1

        cursor = await connection.execute(
            """
            SELECT
                json_extract(key_json, '$.policy_json.privacy_enforcement') AS privacy,
                json_extract(key_json, '$.operational_json.operational_profile.token') AS token,
                COUNT(*) AS count
            FROM initial_context_packages
            WHERE user_id = ?
              AND build_status = 'active'
            GROUP BY privacy, token
            """,
            ("usr_1",),
        )
        counts = {
            (str(row["privacy"]), str(row["token"])): int(row["count"])
            for row in await cursor.fetchall()
        }
        assert counts == {
            ("enforce", "offline"): 2,
            ("off", "offline"): 2,
            ("off", "online"): 2,
        }
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_post_dispatch_stale_cleanup_cannot_stale_completed_same_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        initial_enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        await initial_enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        assert (await worker.run_once()).acked == 1

        original_enqueue = backend.job_tracking_service.enqueue_job
        completed_generation: int | None = None
        completed_result: dict[str, object] | None = None
        completed_error: BaseException | None = None

        async def dispatch_and_complete(
            storage_backend: object,
            stream_name: str,
            job: object,
            **kwargs: object,
        ) -> None:
            nonlocal completed_error, completed_generation, completed_result
            await original_enqueue(
                storage_backend,  # type: ignore[arg-type]
                stream_name,
                job,  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )
            envelope = job.model_dump(mode="json")  # type: ignore[attr-defined]
            completed_generation = int(envelope["payload"]["refresh_generation"])
            try:
                async with bound_test_job_claim(
                    connection,
                    backend,
                    clock,
                    envelope,
                ):
                    result = await worker.process_job(envelope)
            except BaseException as exc:
                completed_error = exc
                raise
            completed_result = result
            assert result["status"] == "refreshed"

        monkeypatch.setattr(
            backend.job_tracking_service,
            "enqueue_job",
            dispatch_and_complete,
        )
        job_id = await InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
            package_repository=InitialContextPackageRepository(connection, clock),
        ).enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            force=True,
        )

        assert job_id is not None
        assert completed_generation is not None
        assert completed_error is None
        assert completed_result is not None
        assert len(completed_result["built_packages"]) == 2  # type: ignore[arg-type]
        rows = await (
            await connection.execute(
                """
                SELECT build_status, refresh_generation
                FROM initial_context_packages
                WHERE user_id = ?
                """,
                ("usr_1",),
            )
        ).fetchall()
        assert len(rows) == 2
        assert {str(row["build_status"]) for row in rows} == {"active"}
        assert {int(row["refresh_generation"]) for row in rows} == {
            completed_generation
        }
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_bulk_refresh_preserves_same_subject_coordinate_variants() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        await ConversationRepository(connection, clock).create_conversation(
            "cnv_2",
            "usr_1",
            "wrk_1",
            "coding_debug",
            "Incognito coordinate variant",
            user_persona_id="persona_jordi",
            platform_id="aurvek",
            character_id="assistant_alpha",
            incognito=True,
        )
        await InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        ).enqueue_refresh(
            user_id="usr_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        assert (await worker.run_once()).acked == 1

        baseline_rows = await (
            await connection.execute(
                """
                SELECT package_key_hash, coordinate_signature_json
                FROM initial_context_packages
                WHERE user_id = ?
                  AND package_kind = 'baseline'
                  AND build_status = 'active'
                ORDER BY package_key_hash
                """,
                ("usr_1",),
            )
        ).fetchall()
        assert len(baseline_rows) == 2
        assert len({str(row["package_key_hash"]) for row in baseline_rows}) == 2
        assert {
            bool(
                json_utils.loads(row["coordinate_signature_json"])["markers_json"][
                    "lifecycle"
                ]["incognito"]
            )
            for row in baseline_rows
        } == {False, True}
        assert (
            await _active_package_count(
                connection,
                package_kind=InitialContextPackageKind.CONVERSATION,
            )
            == 2
        )
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_refresh_enqueue_stales_all_existing_variants_for_source_change() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        await MessageRepository(connection, clock).create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "Estamos preparando variantes del paquete inicial.",
        )
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        for privacy, token in (
            ("off", "offline"),
            ("enforce", "offline"),
        ):
            await enqueuer.enqueue_refresh(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
                reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
                source_message_ids=["msg_1"],
                privacy_enforcement=privacy,
                operational_profile=_operational_snapshot(token),
            )

        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        assert (await worker.run_once()).acked == 1
        assert (await worker.run_once()).acked == 1
        assert (
            await _active_package_count(
                connection,
                package_kind=InitialContextPackageKind.BASELINE,
            )
            == 2
        )
        assert (
            await _active_package_count(
                connection,
                package_kind=InitialContextPackageKind.CONVERSATION,
            )
            == 2
        )

        stale_enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
            package_repository=InitialContextPackageRepository(connection, clock),
        )
        await stale_enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            source_message_ids=["msg_1"],
            privacy_enforcement="off",
            operational_profile=_operational_snapshot("offline"),
            force=True,
        )

        assert (
            await _active_package_count(
                connection,
                package_kind=InitialContextPackageKind.BASELINE,
            )
            == 0
        )
        assert (
            await _active_package_count(
                connection,
                package_kind=InitialContextPackageKind.CONVERSATION,
            )
            == 0
        )
        assert await _package_status_counts(connection) == {"stale": 4}
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_worker_skips_queued_refresh_when_rollout_disabled_and_stales_family() -> (
    None
):
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        enabled_worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        assert (await enabled_worker.run_once()).acked == 1
        assert await _package_status_counts(connection) == {"active": 2}

        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            force=True,
        )
        disabled_worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=replace(
                _settings(), initial_context_package_refresh_enabled=False
            ),
        )
        result = await disabled_worker.run_once()

        assert result.acked == 1
        assert await _package_status_counts(connection) == {"stale": 2}
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_disabled_rollout_old_job_does_not_stale_newer_generation() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        old_job_id = await InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        ).enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        assert old_job_id is not None
        old_run = await backend.job_tracking_service.get_job_run(old_job_id)
        assert old_run is not None
        old_envelope = old_run["recovery_envelope_json"]
        old_generation = int(old_envelope["payload"]["refresh_generation"])
        newer_generation = (
            await backend.job_tracking_service.reserve_icp_refresh_generation("usr_1")
        )
        assert newer_generation > old_generation

        synthetic_envelope = json_utils.loads(
            json_utils.dumps(old_envelope, sort_keys=True)
        )
        synthetic_envelope["job_id"] = "job_newer_generation_probe"
        synthetic_envelope["payload"]["refresh_generation"] = newer_generation
        enabled_worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        async with bound_test_job_claim(
            connection,
            backend,
            clock,
            synthetic_envelope,
        ):
            refreshed = await enabled_worker.process_job(synthetic_envelope)
        assert refreshed["status"] == "refreshed"

        disabled_worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=replace(
                _settings(), initial_context_package_refresh_enabled=False
            ),
        )
        result = await disabled_worker.run_once()

        assert result.acked == 1
        rows = await (
            await connection.execute(
                """
                SELECT package_kind, build_status, refresh_generation
                FROM initial_context_packages
                WHERE user_id = ?
                ORDER BY package_kind
                """,
                ("usr_1",),
            )
        ).fetchall()
        assert len(rows) == 2
        assert {str(row["package_kind"]) for row in rows} == {
            "baseline",
            "conversation",
        }
        assert {str(row["build_status"]) for row in rows} == {"active"}
        assert {int(row["refresh_generation"]) for row in rows} == {newer_generation}
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_worker_stales_previous_coordinate_key_before_rebuild() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        assert (await worker.run_once()).acked == 1

        await ConversationRepository(connection, clock).set_active_space(
            "cnv_1",
            "usr_1",
            "space_changed",
        )
        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.COORDINATE_CHANGE,
            force=True,
        )
        assert (await worker.run_once()).acked == 1

        status_counts = await _package_status_counts(connection)
        assert status_counts["active"] == 3
        assert status_counts["stale"] == 1
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_worker_baseline_build_failure_preserves_previous_active_package() -> (
    None
):
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        assert (await worker.run_once()).acked == 1
        assert await _package_status_counts(connection) == {"active": 2}

        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            package_kind=InitialContextPackageKind.BASELINE,
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
            force=True,
        )
        worker._builder = FailingPackageBuilder("build_baseline_package")  # noqa: SLF001

        result = await worker.run_once()

        assert result.failed == 1
        assert result.acked == 0
        assert await _package_status_counts(connection) == {"active": 2}
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_worker_conversation_build_failure_preserves_previous_active_package() -> (
    None
):
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        assert (await worker.run_once()).acked == 1
        assert await _package_status_counts(connection) == {"active": 2}

        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            package_kind=InitialContextPackageKind.CONVERSATION,
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
            force=True,
        )
        worker._builder = FailingPackageBuilder("build_conversation_package")  # noqa: SLF001

        result = await worker.run_once()

        assert result.failed == 1
        assert result.acked == 0
        assert await _package_status_counts(connection) == {"active": 2}
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_lifecycle_close_deletes_conversation_package_and_stales_baseline() -> (
    None
):
    connection, clock, manifest_loader = await _seed_runtime()
    backend = DurableJobTestBackend(connection, clock, settings=_settings())
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
            job_tracking_service=backend.job_tracking_service,
        )
        await enqueuer.enqueue_refresh(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
            reason=InitialContextPackageRefreshReason.BACKFILL,
        )
        worker = InitialContextPackageWorker(
            backend,
            connection,
            clock,
            manifest_loader,
            settings=_settings(),
        )
        assert (await worker.run_once()).acked == 1

        runtime = SimpleNamespace(
            clock=clock,
            storage_backend=backend,
            database_path=":memory:",
            settings=_settings(),
            llm_client=LLMClient(provider_name="test", providers=[]),
            embedding_index=None,
        )
        await ConversationLifecycleService(runtime).close_conversation(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
            purge=False,
        )

        assert (
            await InitialContextPackageRepository(
                connection,
                clock,
            ).get_latest_for_conversation(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
            )
            is None
        )
        status_counts = await _package_status_counts(connection)
        assert status_counts == {"stale": 1}
    finally:
        await backend.close()
        await connection.close()
