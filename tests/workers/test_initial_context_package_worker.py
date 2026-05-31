"""Tests for prepared initial-context package refresh work."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from atagia.core import json_utils
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.storage_backend import InProcessBackend
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

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


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


async def _seed_runtime() -> tuple[aiosqlite.Connection, FrozenClock, ManifestLoader]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
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
    return {str(row["build_status"]): int(row["count"]) for row in await cursor.fetchall()}


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


@pytest.mark.asyncio
async def test_refresh_enqueuer_coalesces_and_worker_materializes_packages() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = InProcessBackend()
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
        assert conversation_package.blocks_json.recent_verbatim_seed[0]["message_id"] == "msg_1"
        assert await _active_package_count(
            connection,
            package_kind=InitialContextPackageKind.BASELINE,
        ) == 1
        assert await _active_package_count(
            connection,
            package_kind=InitialContextPackageKind.CONVERSATION,
        ) == 1
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_worker_uses_llm_curation_for_background_package_refresh() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = InProcessBackend()
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
        assert conversation_package.source_refs_json["curated_orientation"][0][
            "memory_id"
        ] == "mem_worker"
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_refresh_variants_do_not_coalesce_or_stale_each_other() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = InProcessBackend()
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
async def test_refresh_enqueue_stales_all_existing_variants_for_source_change() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = InProcessBackend()
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
        assert await _active_package_count(
            connection,
            package_kind=InitialContextPackageKind.BASELINE,
        ) == 2
        assert await _active_package_count(
            connection,
            package_kind=InitialContextPackageKind.CONVERSATION,
        ) == 2

        stale_enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
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

        assert await _active_package_count(
            connection,
            package_kind=InitialContextPackageKind.BASELINE,
        ) == 0
        assert await _active_package_count(
            connection,
            package_kind=InitialContextPackageKind.CONVERSATION,
        ) == 0
        assert await _package_status_counts(connection) == {"stale": 4}
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_worker_skips_queued_refresh_when_rollout_disabled_and_stales_family() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = InProcessBackend()
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
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
            settings=replace(_settings(), initial_context_package_refresh_enabled=False),
        )
        result = await disabled_worker.run_once()

        assert result.acked == 1
        assert await _package_status_counts(connection) == {"stale": 2}
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_worker_stales_previous_coordinate_key_before_rebuild() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = InProcessBackend()
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
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
        assert status_counts["active"] == 2
        assert status_counts["stale"] == 2
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_worker_baseline_build_failure_preserves_previous_active_package() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = InProcessBackend()
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
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
async def test_worker_conversation_build_failure_preserves_previous_active_package() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = InProcessBackend()
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
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
async def test_lifecycle_close_deletes_conversation_package_and_stales_baseline() -> None:
    connection, clock, manifest_loader = await _seed_runtime()
    backend = InProcessBackend()
    try:
        enqueuer = InitialContextPackageRefreshEnqueuer(
            storage_backend=backend,
            clock=clock,
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
            artifact_blob_store=None,
        )
        await ConversationLifecycleService(runtime).close_conversation(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
            purge=False,
        )

        assert await InitialContextPackageRepository(
            connection,
            clock,
        ).get_latest_for_conversation(
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
        ) is None
        status_counts = await _package_status_counts(connection)
        assert status_counts == {"stale": 1}
    finally:
        await backend.close()
        await connection.close()
