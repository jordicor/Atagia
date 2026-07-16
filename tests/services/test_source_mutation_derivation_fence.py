"""Revision-fence coverage for authoritative source mutations."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database, open_connection
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.memory_evidence_repository import MemoryEvidenceRepository
from atagia.core.memory_fact_facet_repository import MemoryFactFacetRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
    summary_mirror_id,
)
from atagia.core.storage_backend import InProcessBackend
from atagia.core.summary_repository import SummaryRepository
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.inspector import MemoryInspector
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_jobs import (
    ClaimedJob,
    DurableJobNotification,
    JobEnvelope,
    JobType,
)
from atagia.models.schemas_memory import (
    ExactFacet,
    MemoryEvidenceSupportKind,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
    SummaryViewKind,
)
from atagia.services.admin_rebuild_service import AdminRebuildService
from atagia.services.job_execution_context import (
    bind_job_claim,
    current_derivation_dedupe_scope,
    reset_job_claim,
)
from atagia.services.lifecycle_service import (
    DELETE_CONVERSATION_CONFIRMATION,
    HARD_DELETE_MEMORY_CONFIRMATION,
    ConversationLifecycleService,
)
from atagia.services.errors import MemoryProvenanceRepairRequiredError
from atagia.services.sidecar_service import SidecarService
from atagia.services.worker_effect_fence import (
    StaleJobEffectFenceError,
    WorkerEffectFence,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)
USER_ID = "usr_1"
CONVERSATION_ID = "cnv_1"
MEMORY_ID = "mem_1"
CITY_MEMORY_ID = "mem_city"


def _candidate_settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        fact_facet_retrieval_enabled=True,
        fact_facet_structured_only=True,
        verbatim_evidence_search_enabled=False,
    )


def _city_plan(query: str) -> RetrievalPlan:
    return RetrievalPlan(
        original_query=f"What is my current city: {query}?",
        assistant_mode_id="coding_debug",
        workspace_id=None,
        conversation_id=CONVERSATION_ID,
        fts_queries=[query],
        sub_query_plans=[
            PlannedSubQuery(
                text=query,
                sparse_phrase=query,
                must_keep_terms=[query],
                fts_queries=[query],
            )
        ],
        query_type="slot_fill",
        scope_filter=[MemoryScope.CONVERSATION],
        status_filter=[MemoryStatus.ACTIVE],
        vector_limit=0,
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=1,
        retrieval_levels=[0],
        exact_recall_mode=True,
        exact_facets=[ExactFacet.LOCATION],
        answer_shape="single_fact",
        coverage_mode="current_state",
        source_precision="required",
    )


class _NoopEmbeddingIndex:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete(self, memory_id: str) -> None:
        self.deleted.append(memory_id)


class _Runtime:
    def __init__(
        self,
        database_path: str,
        clock: FrozenClock,
        storage_backend: InProcessBackend,
    ) -> None:
        self.database_path = database_path
        self.clock = clock
        self.storage_backend = storage_backend
        self.embedding_index = _NoopEmbeddingIndex()
        self.llm_client = None
        self.settings = SimpleNamespace(
            erasure_purge_streams=False,
            storage_backend="inprocess",
        )

    async def open_connection(self) -> aiosqlite.Connection:
        return await open_connection(self.database_path)


class _PauseAfterImmediate:
    """Pause after the lifecycle transaction has acquired the writer lock."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self.transaction_started = asyncio.Event()
        self.resume_mutation = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if " ".join(sql.split()).upper() == "BEGIN IMMEDIATE":
            result = await self._connection.execute(sql, *args, **kwargs)
            self.transaction_started.set()
            await asyncio.wait_for(self.resume_mutation.wait(), timeout=5.0)
            return result
        return await self._connection.execute(sql, *args, **kwargs)


class _ObserveFirstWrite:
    """Expose when a competing connection attempts its first SQLite write."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self.write_attempted = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        normalized = " ".join(sql.split()).upper()
        if normalized.startswith(("INSERT ", "UPDATE ", "DELETE ", "BEGIN ")):
            self.write_attempted.set()
        return await self._connection.execute(sql, *args, **kwargs)


async def _seed_database(
    database_path: str,
) -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(
        connection,
        ManifestLoader(MANIFESTS_DIR).load_all(),
        clock,
    )
    await UserRepository(connection, clock).create_user(USER_ID)
    await ConversationRepository(connection, clock).create_conversation(
        CONVERSATION_ID,
        USER_ID,
        None,
        "coding_debug",
        "Fence source",
    )
    message = await MessageRepository(connection, clock).create_message(
        "msg_1",
        CONVERSATION_ID,
        "user",
        1,
        "The canonical source message.",
    )
    await MemoryObjectRepository(connection, clock).create_memory_object(
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Canonical source memory",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        memory_id=MEMORY_ID,
        payload={"source_message_ids": [str(message["id"])]},
    )
    return connection, clock


async def _seed_city_retrieval_surfaces(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
) -> None:
    source = await MessageRepository(connection, clock).create_message(
        "msg_city",
        CONVERSATION_ID,
        "user",
        2,
        "My current city is Paris.",
        occurred_at="2026-07-13T11:00:00+00:00",
    )
    await MemoryObjectRepository(connection, clock).create_memory_object(
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="Paris",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.91,
        privacy_level=0,
        memory_id=CITY_MEMORY_ID,
        extraction_hash="city-extraction-hash",
        payload={"source_message_ids": [str(source["id"])]},
        scope_canonical=MemoryScope.CHAT.value,
        language_codes=["en"],
        commit=False,
    )
    packet = await MemoryEvidenceRepository(
        connection,
        clock,
    ).create_support_edge_with_spans(
        user_id=USER_ID,
        memory_id=CITY_MEMORY_ID,
        support_kind=MemoryEvidenceSupportKind.DIRECT,
        confidence=0.91,
        spans=[
            {
                "span_role": "source",
                "message_id": str(source["id"]),
                "conversation_id": CONVERSATION_ID,
                "quote_text": "My current city is Paris.",
                "occurred_at": "2026-07-13T11:00:00+00:00",
            }
        ],
        commit=False,
    )
    await MemoryFactFacetRepository(connection, clock).upsert_fact_facet(
        user_id=USER_ID,
        memory_id=CITY_MEMORY_ID,
        source_span_id=str(packet["spans"][0]["id"]),
        source_message_id=str(source["id"]),
        conversation_id=CONVERSATION_ID,
        subject_surface=USER_ID,
        surface_class="structured",
        facet_label="location.current_city",
        value_text="Paris",
        support_kind=MemoryEvidenceSupportKind.DIRECT.value,
        observed_at="2026-07-13T11:00:00+00:00",
        current_state=True,
        language_code="en",
        confidence=0.91,
        commit=False,
    )
    await connection.commit()


async def _seed_transitive_city_summaries(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
) -> list[str]:
    summaries = SummaryRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    created_at = "2026-07-13T11:05:00+00:00"
    await summaries.create_summary(
        USER_ID,
        {
            "id": "sum_city_episode",
            "conversation_id": CONVERSATION_ID,
            "workspace_id": None,
            "source_message_start_seq": None,
            "source_message_end_seq": None,
            "summary_kind": SummaryViewKind.EPISODE.value,
            "hierarchy_level": 1,
            "summary_text": "Paris is the user's current city.",
            "source_object_ids_json": [CITY_MEMORY_ID],
            "maya_score": 1.5,
            "model": "test-summary-model",
            "created_at": created_at,
        },
        commit=False,
    )
    await memories.upsert_summary_mirror(
        user_id=USER_ID,
        summary_view_id="sum_city_episode",
        summary_kind=SummaryViewKind.EPISODE,
        hierarchy_level=1,
        summary_text="Paris is the user's current city.",
        source_object_ids=[CITY_MEMORY_ID],
        created_at=created_at,
        scope=MemoryScope.CONVERSATION,
        conversation_id=CONVERSATION_ID,
        assistant_mode_id="coding_debug",
        scope_canonical=MemoryScope.CHAT.value,
        language_codes=["en"],
        commit=False,
    )
    first_mirror_id = summary_mirror_id("sum_city_episode")
    await summaries.create_summary(
        USER_ID,
        {
            "id": "sum_city_profile",
            "conversation_id": None,
            "workspace_id": None,
            "source_message_start_seq": None,
            "source_message_end_seq": None,
            "summary_kind": SummaryViewKind.THEMATIC_PROFILE.value,
            "hierarchy_level": 2,
            "summary_text": "The profile still says the user lives in Paris.",
            "source_object_ids_json": [first_mirror_id],
            "maya_score": 1.5,
            "model": "test-summary-model",
            "created_at": created_at,
        },
        commit=False,
    )
    await memories.upsert_summary_mirror(
        user_id=USER_ID,
        summary_view_id="sum_city_profile",
        summary_kind=SummaryViewKind.THEMATIC_PROFILE,
        hierarchy_level=2,
        summary_text="The profile still says the user lives in Paris.",
        source_object_ids=[first_mirror_id],
        created_at=created_at,
        scope=MemoryScope.USER,
        assistant_mode_id="coding_debug",
        scope_canonical=MemoryScope.USER.value,
        language_codes=["en"],
        commit=False,
    )
    await connection.commit()
    return [
        first_mirror_id,
        summary_mirror_id("sum_city_profile"),
    ]


def _job(job_id: str) -> JobEnvelope:
    return JobEnvelope(
        job_id=job_id,
        job_type=JobType.RUN_EVALUATION,
        user_id=USER_ID,
        payload={"metrics": ["system"]},
    )


async def _create_and_claim(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
    *,
    job_id: str,
) -> ClaimedJob:
    repository = JobRunRepository(connection, clock)
    await repository.create_durable_job(
        stream_name="atagia:evaluate",
        target_backend="inprocess",
        envelope=_job(job_id),
        source_token_estimate=None,
        size_bucket=None,
    )
    rows = await repository.claim_dispatchable_jobs(
        target_backend="inprocess",
        limit=20,
        visibility_seconds=30,
    )
    row = next(item for item in rows if item["job_id"] == job_id)
    claim = await repository.claim_notification(
        f"delivery-{job_id}",
        DurableJobNotification(
            job_id=job_id,
            dispatch_token=str(row["dispatch_token"]),
            lifecycle_epoch=str(row["lifecycle_epoch"]),
            lifecycle_cleanup_key=str(row["lifecycle_cleanup_key"]),
        ),
        owner_id=f"owner-{job_id}",
        lease_seconds=30,
    )
    assert claim is not None
    return claim


def _claim_scope(claim: ClaimedJob) -> str:
    token = bind_job_claim(claim)
    try:
        return current_derivation_dedupe_scope(
            user_id=claim.envelope.user_id,
            job_id=claim.envelope.job_id,
        )
    finally:
        reset_job_claim(token)


async def _assert_stale_claim_and_new_revision_can_publish(
    *,
    job_connection: aiosqlite.Connection,
    effect_connection: aiosqlite.Connection,
    clock: FrozenClock,
    storage_backend: InProcessBackend,
    stale_claim: ClaimedJob,
    case: str,
) -> ClaimedJob:
    identity = await UserLifecycleRepository(
        job_connection,
        clock,
    ).get_active_identity(USER_ID)
    assert identity is not None
    assert identity.derivation_revision > stale_claim.derivation_revision

    with pytest.raises(StaleJobEffectFenceError):
        async with WorkerEffectFence(effect_connection, clock).activate(stale_claim):
            await WorkspaceRepository(effect_connection, clock).create_workspace(
                f"wrk_stale_{case}",
                USER_ID,
                "Stale derived effect",
            )

    fresh_claim = await _create_and_claim(
        job_connection,
        clock,
        job_id=f"job_fresh_{case}",
    )
    assert fresh_claim.derivation_revision == identity.derivation_revision
    async with WorkerEffectFence(effect_connection, clock).activate(fresh_claim):
        await WorkspaceRepository(effect_connection, clock).create_workspace(
            f"wrk_fresh_{case}",
            USER_ID,
            "Rebuilt derived effect",
        )

    stale_dedupe_key = f"rebuild:{USER_ID}:{_claim_scope(stale_claim)}"
    fresh_dedupe_key = f"rebuild:{USER_ID}:{_claim_scope(fresh_claim)}"
    assert stale_dedupe_key != fresh_dedupe_key
    await storage_backend.force_dedupe(stale_dedupe_key, ttl_seconds=60)
    assert await storage_backend.remember_dedupe(
        fresh_dedupe_key,
        ttl_seconds=60,
    )
    return fresh_claim


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "archive_conversation",
        "edit_memory",
        "archive_memory",
        "hard_delete_memory",
    ],
)
async def test_single_transaction_source_mutation_fences_user_wide_job(
    mutation: str,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / f"{mutation}.db")
    job_connection, clock = await _seed_database(database_path)
    effect_connection = await open_connection(database_path)
    backend = InProcessBackend()
    runtime = _Runtime(database_path, clock, backend)
    try:
        stale_claim = await _create_and_claim(
            job_connection,
            clock,
            job_id=f"job_stale_{mutation}",
        )
        service = ConversationLifecycleService(runtime)
        if mutation == "archive_conversation":
            await service.archive_conversation(
                job_connection,
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
            )
        elif mutation == "edit_memory":
            await service.edit_memory(
                job_connection,
                user_id=USER_ID,
                memory_id=MEMORY_ID,
                new_text="Edited canonical source memory",
            )
        elif mutation == "archive_memory":
            await service.delete_memory(
                job_connection,
                user_id=USER_ID,
                memory_id=MEMORY_ID,
            )
        else:
            await service.delete_memory(
                job_connection,
                user_id=USER_ID,
                memory_id=MEMORY_ID,
                hard=True,
                confirmation=HARD_DELETE_MEMORY_CONFIRMATION,
            )

        await _assert_stale_claim_and_new_revision_can_publish(
            job_connection=job_connection,
            effect_connection=effect_connection,
            clock=clock,
            storage_backend=backend,
            stale_claim=stale_claim,
            case=mutation,
        )
    finally:
        await backend.close()
        await effect_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
async def test_delete_is_one_transaction_and_blocks_intervening_job_claim(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "delete-one-transaction.db")
    mutation_connection, clock = await _seed_database(database_path)
    raw_claim_connection = await open_connection(database_path)
    claim_connection = _ObserveFirstWrite(raw_claim_connection)
    backend = InProcessBackend()
    runtime = _Runtime(database_path, clock, backend)
    pausing_connection = _PauseAfterImmediate(mutation_connection)
    delete_task: asyncio.Task[Any] | None = None
    claim_task: asyncio.Task[ClaimedJob] | None = None
    try:
        delete_task = asyncio.create_task(
            ConversationLifecycleService(runtime).delete_conversation(
                pausing_connection,
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
                confirmation=DELETE_CONVERSATION_CONFIRMATION,
            )
        )
        await asyncio.wait_for(
            pausing_connection.transaction_started.wait(), timeout=5.0
        )
        claim_task = asyncio.create_task(
            _create_and_claim(
                claim_connection,
                clock,
                job_id="job_after_atomic_delete",
            )
        )
        await asyncio.wait_for(claim_connection.write_attempted.wait(), timeout=5.0)
        assert not claim_task.done()

        pausing_connection.resume_mutation.set()
        await asyncio.wait_for(delete_task, timeout=5.0)
        delete_task = None
        claim = await asyncio.wait_for(claim_task, timeout=5.0)
        claim_task = None

        final_identity = await UserLifecycleRepository(
            claim_connection,
            clock,
        ).get_active_identity(USER_ID)
        assert final_identity is not None
        assert claim.derivation_revision == final_identity.derivation_revision
    finally:
        if delete_task is not None:
            pausing_connection.resume_mutation.set()
            delete_task.cancel()
            await asyncio.gather(delete_task, return_exceptions=True)
        if claim_task is not None:
            claim_task.cancel()
            await asyncio.gather(claim_task, return_exceptions=True)
        await backend.close()
        await raw_claim_connection.close()
        await mutation_connection.close()


@pytest.mark.asyncio
async def test_resumed_pending_delete_refences_existing_claim(tmp_path: Path) -> None:
    database_path = str(tmp_path / "resumed-pending-delete.db")
    mutation_connection, clock = await _seed_database(database_path)
    effect_connection = await open_connection(database_path)
    backend = InProcessBackend()
    runtime = _Runtime(database_path, clock, backend)
    try:
        await mutation_connection.execute(
            "UPDATE conversations SET status = 'pending_deletion' WHERE id = ?",
            (CONVERSATION_ID,),
        )
        await mutation_connection.commit()
        stale_claim = await _create_and_claim(
            mutation_connection,
            clock,
            job_id="job_stale_resumed_delete",
        )

        await ConversationLifecycleService(runtime)._purge_pending_conversation(
            mutation_connection,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
        )

        await _assert_stale_claim_and_new_revision_can_publish(
            job_connection=mutation_connection,
            effect_connection=effect_connection,
            clock=clock,
            storage_backend=backend,
            stale_claim=stale_claim,
            case="resumed_delete",
        )
    finally:
        await backend.close()
        await effect_connection.close()
        await mutation_connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["preferences", "incognito"])
async def test_sidecar_source_policy_mutation_fences_user_wide_job(
    mutation: str,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / f"sidecar-{mutation}.db")
    job_connection, clock = await _seed_database(database_path)
    effect_connection = await open_connection(database_path)
    backend = InProcessBackend()
    runtime = _Runtime(database_path, clock, backend)
    try:
        stale_claim = await _create_and_claim(
            job_connection,
            clock,
            job_id=f"job_stale_{mutation}",
        )
        sidecar = SidecarService(runtime)
        if mutation == "preferences":
            await sidecar.set_memory_preferences(
                USER_ID,
                remember_across_chats=False,
            )
        else:
            await sidecar.set_conversation_incognito(
                USER_ID,
                CONVERSATION_ID,
                True,
            )

        await _assert_stale_claim_and_new_revision_can_publish(
            job_connection=job_connection,
            effect_connection=effect_connection,
            clock=clock,
            storage_backend=backend,
            stale_claim=stale_claim,
            case=mutation,
        )
    finally:
        await backend.close()
        await effect_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["conversation", "user"])
async def test_admin_purge_fences_user_wide_job(scope: str, tmp_path: Path) -> None:
    database_path = str(tmp_path / f"admin-purge-{scope}.db")
    job_connection, clock = await _seed_database(database_path)
    effect_connection = await open_connection(database_path)
    backend = InProcessBackend()
    try:
        stale_claim = await _create_and_claim(
            job_connection,
            clock,
            job_id=f"job_stale_admin_{scope}",
        )
        service = AdminRebuildService(
            connection=job_connection,
            llm_client=None,  # type: ignore[arg-type]
            embedding_index=None,
            clock=clock,
            manifest_loader=ManifestLoader(MANIFESTS_DIR),
            settings=SimpleNamespace(),  # type: ignore[arg-type]
        )
        if scope == "conversation":
            await service._purge_conversation_state(USER_ID, CONVERSATION_ID)
        else:
            await service._purge_user_state(USER_ID)

        await _assert_stale_claim_and_new_revision_can_publish(
            job_connection=job_connection,
            effect_connection=effect_connection,
            clock=clock,
            storage_backend=backend,
            stale_claim=stale_claim,
            case=f"admin_{scope}",
        )
    finally:
        await backend.close()
        await effect_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
async def test_coordinate_correction_bumps_revision_and_fences_old_worker_claim(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "coordinate-correction-fence.db")
    job_connection, clock = await _seed_database(database_path)
    effect_connection = await open_connection(database_path)
    backend = InProcessBackend()
    try:
        stale_claim = await _create_and_claim(
            job_connection,
            clock,
            job_id="job_stale_coordinate_correction",
        )
        corrected = await MemoryInspector(
            job_connection,
            clock,
        ).correct_memory_coordinates(
            MEMORY_ID,
            USER_ID,
            admin_user_id="admin_test",
            updates={"presence_cluster_id": "cluster_corrected"},
            reason="correct imported coordinate",
        )
        assert corrected is not None
        assert (
            corrected["coordinates"]["presence"]["presence_cluster_id"]
            == "cluster_corrected"
        )

        await _assert_stale_claim_and_new_revision_can_publish(
            job_connection=job_connection,
            effect_connection=effect_connection,
            clock=clock,
            storage_backend=backend,
            stale_claim=stale_claim,
            case="coordinate_correction",
        )
    finally:
        await backend.close()
        await effect_connection.close()
        await job_connection.close()


@pytest.mark.asyncio
async def test_normalized_conversation_provenance_is_exhaustive_but_not_transitive(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "normalized-provenance.db")
    connection, clock = await _seed_database(database_path)
    backend = InProcessBackend()
    runtime = _Runtime(database_path, clock, backend)
    timestamp = clock.now().isoformat()
    try:
        memories = MemoryObjectRepository(connection, clock)
        await memories.create_memory_object(
            user_id=USER_ID,
            conversation_id=None,
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.USER,
            canonical_text="Normalized evidence-backed memory",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_normalized",
            extraction_hash="normalized-extraction-hash",
            payload={},
        )
        await memories.create_memory_object(
            user_id=USER_ID,
            conversation_id=None,
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.BELIEF,
            scope=MemoryScope.USER,
            canonical_text="Independent manually curated truth",
            source_kind=MemorySourceKind.VERBATIM,
            confidence=1.0,
            privacy_level=0,
            memory_id="mem_manual",
            payload={"writer_kind": "manual"},
        )
        await connection.execute(
            """
            INSERT INTO memory_support_edges(
                id, user_id, memory_id, confidence, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("mse_1", USER_ID, "mem_normalized", 0.9, timestamp, timestamp),
        )
        await connection.execute(
            """
            INSERT INTO memory_evidence_spans(
                id, user_id, support_edge_id, memory_id, conversation_id,
                message_id, span_role, quote_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'source', ?, ?, ?)
            """,
            (
                "mes_1",
                USER_ID,
                "mse_1",
                "mem_normalized",
                CONVERSATION_ID,
                "msg_1",
                "The canonical source message.",
                timestamp,
                timestamp,
            ),
        )
        await connection.execute(
            """
            INSERT INTO memory_fact_facets(
                id, user_id, conversation_id, memory_id, source_message_id,
                source_span_id, source_hash, subject_surface, facet_label,
                value_text, value_norm_key, assertion_kind, support_kind,
                observed_at, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "mff_1",
                USER_ID,
                CONVERSATION_ID,
                "mem_normalized",
                "msg_1",
                "mes_1",
                "source-hash",
                "user",
                "preference",
                "canonical",
                "canonical",
                "state",
                "direct",
                timestamp,
                0.9,
                timestamp,
            ),
        )
        await connection.execute(
            """
            INSERT INTO memory_links(
                id, user_id, src_memory_id, dst_memory_id, relation_type,
                weight, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, 'supports', 1.0, '{}', ?)
            """,
            ("ml_manual", USER_ID, "mem_manual", "mem_normalized", timestamp),
        )
        await connection.commit()

        service = ConversationLifecycleService(runtime)
        affected = await service._conversation_affected_memory_ids(
            connection,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
        )
        assert "mem_normalized" in affected
        assert "mem_manual" not in affected

        await service.archive_conversation(
            connection,
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
        )
        normalized = await memories.get_memory_object("mem_normalized", USER_ID)
        manual = await memories.get_memory_object("mem_manual", USER_ID)
        assert normalized is not None and normalized["status"] == "archived"
        assert manual is not None and manual["status"] == "active"
        cursor = await connection.execute(
            """
            SELECT source_message_id, retired_memory_id
            FROM memory_extraction_suppressions
            WHERE user_id = ?
              AND retired_memory_id = ?
            """,
            (USER_ID, "mem_normalized"),
        )
        assert [dict(row) for row in await cursor.fetchall()] == [
            {
                "source_message_id": "msg_1",
                "retired_memory_id": "mem_normalized",
            }
        ]
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_memory_edit_removes_old_fact_and_evidence_retrieval_surfaces(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "edit-retires-retrieval-surfaces.db")
    connection, clock = await _seed_database(database_path)
    backend = InProcessBackend()
    runtime = _Runtime(database_path, clock, backend)
    search = CandidateSearch(
        connection,
        clock,
        settings=_candidate_settings(),
    )
    try:
        await _seed_city_retrieval_surfaces(connection, clock)
        summary_mirror_ids = await _seed_transitive_city_summaries(connection, clock)
        await MemoryObjectRepository(connection, clock).create_memory_object(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Unrelated outcome",
            source_kind=MemorySourceKind.VERBATIM,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_unrelated_outcome",
            payload={"writer_kind": "manual"},
            commit=False,
        )
        timestamp = clock.now().isoformat()
        await connection.execute(
            """
            INSERT INTO consequence_chains(
                id,
                user_id,
                conversation_id,
                assistant_mode_id,
                action_memory_id,
                outcome_memory_id,
                confidence,
                status,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cch_unrelated",
                USER_ID,
                CONVERSATION_ID,
                "coding_debug",
                MEMORY_ID,
                "mem_unrelated_outcome",
                0.8,
                "active",
                timestamp,
                timestamp,
            ),
        )
        await connection.commit()
        before = await search.search(_city_plan("Paris"), USER_ID)
        old_facts = [
            row for row in before if row.get("fact_facet_memory_id") == CITY_MEMORY_ID
        ]
        assert len(old_facts) == 1
        assert old_facts[0]["payload_json"]["value_text"] == "Paris"
        assert old_facts[0]["evidence_packets"][0]["spans"][0]["quote_text"] == (
            "My current city is Paris."
        )

        await ConversationLifecycleService(runtime).edit_memory(
            connection,
            user_id=USER_ID,
            memory_id=CITY_MEMORY_ID,
            new_text="Madrid",
        )

        stale_results = await search.search(_city_plan("Paris"), USER_ID)
        assert all(
            row.get("id") != CITY_MEMORY_ID
            and row.get("fact_facet_memory_id") != CITY_MEMORY_ID
            for row in stale_results
        )
        current_results = await search.search(_city_plan("Madrid"), USER_ID)
        current = [row for row in current_results if row.get("id") == CITY_MEMORY_ID]
        assert len(current) == 1
        assert current[0]["canonical_text"] == "Madrid"
        assert current[0].get("evidence_packets") in (None, [])
        for summary_id in ("sum_city_episode", "sum_city_profile"):
            cursor = await connection.execute(
                "SELECT 1 FROM summary_views WHERE user_id = ? AND id = ?",
                (USER_ID, summary_id),
            )
            assert await cursor.fetchone() is None
        for mirror_id in summary_mirror_ids:
            assert (
                await MemoryObjectRepository(connection, clock).get_memory_object(
                    mirror_id,
                    USER_ID,
                )
                is None
            )
        assert {CITY_MEMORY_ID, *summary_mirror_ids}.issubset(
            set(runtime.embedding_index.deleted)
        )
        cursor = await connection.execute(
            "SELECT 1 FROM consequence_chains WHERE id = ?",
            ("cch_unrelated",),
        )
        assert await cursor.fetchone() is not None
        assert (
            await MemoryEvidenceRepository(
                connection,
                clock,
            ).list_packets_for_memory_ids(
                user_id=USER_ID,
                memory_ids=[CITY_MEMORY_ID],
            )
            == {}
        )
        for table in (
            "memory_fact_facets",
            "memory_evidence_spans",
            "memory_support_edges",
        ):
            cursor = await connection.execute(
                f"SELECT COUNT(*) AS row_count FROM {table} WHERE memory_id = ?",
                (CITY_MEMORY_ID,),
            )
            assert int((await cursor.fetchone())["row_count"]) == 0
        cursor = await connection.execute(
            """
            SELECT previous_text, new_text
            FROM memory_edit_history
            WHERE memory_id = ?
            """,
            (CITY_MEMORY_ID,),
        )
        history = await cursor.fetchone()
        assert history is not None
        assert history["previous_text"] == "Paris"
        assert history["new_text"] == "Madrid"
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
async def test_memory_edit_fails_closed_when_extracted_source_provenance_is_missing(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "edit-missing-source-provenance.db")
    connection, clock = await _seed_database(database_path)
    backend = InProcessBackend()
    runtime = _Runtime(database_path, clock, backend)
    try:
        await MemoryObjectRepository(connection, clock).create_memory_object(
            user_id=USER_ID,
            conversation_id=None,
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.USER,
            canonical_text="Paris",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_missing_provenance",
            extraction_hash="legacy-source-less-hash",
            payload={},
        )
        lifecycle_before = await UserLifecycleRepository(
            connection,
            clock,
        ).get_active_identity(USER_ID)
        assert lifecycle_before is not None

        with pytest.raises(
            MemoryProvenanceRepairRequiredError,
            match="source provenance",
        ):
            await ConversationLifecycleService(runtime).edit_memory(
                connection,
                user_id=USER_ID,
                memory_id="mem_missing_provenance",
                new_text="Madrid",
            )

        stored = await MemoryObjectRepository(connection, clock).get_memory_object(
            "mem_missing_provenance",
            USER_ID,
        )
        assert stored is not None
        assert stored["canonical_text"] == "Paris"
        assert stored["extraction_hash"] == "legacy-source-less-hash"
        lifecycle_after = await UserLifecycleRepository(
            connection,
            clock,
        ).get_active_identity(USER_ID)
        assert lifecycle_after == lifecycle_before
        cursor = await connection.execute(
            """
            SELECT COUNT(*) AS row_count
            FROM memory_edit_history
            WHERE memory_id = ?
            """,
            ("mem_missing_provenance",),
        )
        assert int((await cursor.fetchone())["row_count"]) == 0
        cursor = await connection.execute(
            """
            SELECT COUNT(*) AS row_count
            FROM memory_extraction_suppressions
            WHERE retired_memory_id = ?
            """,
            ("mem_missing_provenance",),
        )
        assert int((await cursor.fetchone())["row_count"]) == 0
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["archive_memory", "hard_delete_memory", "archive_conversation"],
)
async def test_retiring_source_fails_closed_when_extracted_provenance_is_missing(
    mutation: str,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / f"{mutation}-missing-source-provenance.db")
    connection, clock = await _seed_database(database_path)
    backend = InProcessBackend()
    runtime = _Runtime(database_path, clock, backend)
    try:
        await MemoryObjectRepository(connection, clock).create_memory_object(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="Legacy source-less value",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_missing_retirement_provenance",
            extraction_hash="legacy-retirement-source-less-hash",
            payload={},
        )
        lifecycle_before = await UserLifecycleRepository(
            connection,
            clock,
        ).get_active_identity(USER_ID)
        assert lifecycle_before is not None
        service = ConversationLifecycleService(runtime)

        with pytest.raises(
            MemoryProvenanceRepairRequiredError,
            match="source provenance",
        ):
            if mutation == "archive_memory":
                await service.delete_memory(
                    connection,
                    user_id=USER_ID,
                    memory_id="mem_missing_retirement_provenance",
                )
            elif mutation == "hard_delete_memory":
                await service.delete_memory(
                    connection,
                    user_id=USER_ID,
                    memory_id="mem_missing_retirement_provenance",
                    hard=True,
                    confirmation=HARD_DELETE_MEMORY_CONFIRMATION,
                )
            else:
                await service.archive_conversation(
                    connection,
                    user_id=USER_ID,
                    conversation_id=CONVERSATION_ID,
                )

        stored = await MemoryObjectRepository(connection, clock).get_memory_object(
            "mem_missing_retirement_provenance",
            USER_ID,
        )
        assert stored is not None
        assert stored["status"] == MemoryStatus.ACTIVE.value
        conversation = await ConversationRepository(
            connection,
            clock,
        ).get_conversation(CONVERSATION_ID, USER_ID)
        assert conversation is not None
        assert conversation["status"] == "active"
        lifecycle_after = await UserLifecycleRepository(
            connection,
            clock,
        ).get_active_identity(USER_ID)
        assert lifecycle_after == lifecycle_before
    finally:
        await backend.close()
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "archive_memory",
        "hard_delete_memory",
        "archive_conversation",
        "delete_conversation",
    ],
)
async def test_archive_and_delete_hide_old_fact_surfaces_from_active_retrieval(
    mutation: str,
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / f"{mutation}-retires-retrieval-surfaces.db")
    connection, clock = await _seed_database(database_path)
    backend = InProcessBackend()
    runtime = _Runtime(database_path, clock, backend)
    search = CandidateSearch(
        connection,
        clock,
        settings=_candidate_settings(),
    )
    try:
        await _seed_city_retrieval_surfaces(connection, clock)
        summary_mirror_ids = await _seed_transitive_city_summaries(connection, clock)
        assert any(
            row.get("fact_facet_memory_id") == CITY_MEMORY_ID
            for row in await search.search(_city_plan("Paris"), USER_ID)
        )

        service = ConversationLifecycleService(runtime)
        if mutation == "archive_memory":
            await service.delete_memory(
                connection,
                user_id=USER_ID,
                memory_id=CITY_MEMORY_ID,
            )
        elif mutation == "hard_delete_memory":
            await service.delete_memory(
                connection,
                user_id=USER_ID,
                memory_id=CITY_MEMORY_ID,
                hard=True,
                confirmation=HARD_DELETE_MEMORY_CONFIRMATION,
            )
        elif mutation == "archive_conversation":
            await service.archive_conversation(
                connection,
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
            )
        else:
            await service.delete_conversation(
                connection,
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
                confirmation=DELETE_CONVERSATION_CONFIRMATION,
            )

        active_results = await search.search(_city_plan("Paris"), USER_ID)
        assert all(
            row.get("id") != CITY_MEMORY_ID
            and row.get("fact_facet_memory_id") != CITY_MEMORY_ID
            for row in active_results
        )
        for mirror_id in summary_mirror_ids:
            assert (
                await MemoryObjectRepository(connection, clock).get_memory_object(
                    mirror_id,
                    USER_ID,
                )
                is None
            )
        assert {CITY_MEMORY_ID, *summary_mirror_ids}.issubset(
            set(runtime.embedding_index.deleted)
        )
    finally:
        await backend.close()
        await connection.close()
