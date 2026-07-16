"""End-to-end durability tests for authoritative selected transcripts."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
import hashlib
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from atagia.core import json_utils
from atagia.app import AppRuntime, initialize_runtime
from atagia.core.admin_maintenance_repository import (
    AdminMaintenanceRepository,
    admin_maintenance_operation,
)
from atagia.core.artifact_payload_repository import ArtifactPayloadRepository
from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.config import Settings
from atagia.core.clock import FrozenClock
from atagia.core.entity_graph_repository import EntityGraphRepository
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    summary_mirror_id,
)
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.core.transcript_rebuild_repository import TranscriptRebuildRepository
from atagia.models.schemas_api import (
    ReplaceSelectedTranscriptRequest,
    SelectedTranscriptMessage,
)
from atagia.models.schemas_jobs import ClaimedJob, JobEnvelope, JobType
from atagia.models.schemas_memory import (
    ExtractedEvidence,
    ExtractionConversationContext,
    ExtractionResult,
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    SummaryViewKind,
)
from atagia.services.errors import (
    MessageIdConflictError,
    TranscriptRebuildInProgressError,
    TranscriptSelectionConflictError,
)
from atagia.services.admin_rebuild_service import AdminRebuildService
from atagia.services.conversation_activity_service import ConversationActivityService
from atagia.services.selected_transcript_service import SelectedTranscriptService
from atagia.services.worker_job_lease import JobLeaseLostError
from atagia.transport_ids import encode_path_id
from atagia.workers.ingest_worker import IngestWorker
from atagia.workers.transcript_rebuild_worker import TranscriptRebuildWorker

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)

USER_ID = "usr_selected"
CONVERSATION_ID = "cnv/selected?1"
PLATFORM_ID = "openclaw"


class _ObserveImmediate:
    """Signal immediately before a connection waits for its write lock."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self.immediate_attempted = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if " ".join(sql.split()).upper() == "BEGIN IMMEDIATE":
            self.immediate_attempted.set()
        return await self._connection.execute(sql, *args, **kwargs)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "selected-transcript.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        llm_forced_global_model="openai/test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        lifecycle_worker_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _runtime(tmp_path: Path) -> AppRuntime:
    runtime = await initialize_runtime(_settings(tmp_path))
    # These service tests inspect the durable hand-off without starting worker
    # tasks. The production contract still requires workers to be enabled.
    runtime.settings = replace(runtime.settings, workers_enabled=True)
    return runtime


async def _seed_conversation(runtime: AppRuntime) -> None:
    connection = await runtime.open_connection()
    try:
        await UserRepository(connection, runtime.clock).create_user(USER_ID)
        await ConversationRepository(connection, runtime.clock).create_conversation(
            CONVERSATION_ID,
            USER_ID,
            None,
            "general_qa",
            "Selected branch",
            platform_id=PLATFORM_ID,
        )
        messages = MessageRepository(connection, runtime.clock)
        for message_id, role, seq, text, occurred_at in (
            ("msg_keep_u", "user", 1, "keep user", "2026-07-12T12:00:00+00:00"),
            (
                "msg_keep_a",
                "assistant",
                2,
                "keep assistant",
                "2026-07-12T12:01:00+00:00",
            ),
            ("msg_old_u", "user", 3, "abandoned user", "2026-07-12T12:09:00+00:00"),
            (
                "msg_old_a",
                "assistant",
                4,
                "abandoned assistant",
                "2026-07-12T12:10:00+00:00",
            ),
        ):
            await messages.create_message(
                message_id,
                CONVERSATION_ID,
                role,
                seq,
                text,
                occurred_at=occurred_at,
            )
    finally:
        await connection.close()


def _request(
    *,
    operation_id: str = "host/op?1",
    selection_epoch: int = 1,
    new_assistant_text: str = "selected assistant",
) -> ReplaceSelectedTranscriptRequest:
    return ReplaceSelectedTranscriptRequest(
        user_id=USER_ID,
        platform_id=PLATFORM_ID,
        operation_id=operation_id,
        selection_epoch=selection_epoch,
        mutation_kind="regeneration",
        retained_cutoff_message_id="msg_keep_a",
        messages=[
            SelectedTranscriptMessage(
                message_id="msg_keep_u",
                host_message_id="host_keep_u",
                generation_id="generation_1",
                source_namespace="openclaw:selected",
                source_seq=1,
                role="user",
                text="keep user",
                occurred_at="2026-07-12T12:00:00+00:00",
            ),
            SelectedTranscriptMessage(
                message_id="msg_keep_a",
                host_message_id="host_keep_a",
                generation_id="generation_1",
                source_namespace="openclaw:selected",
                source_seq=2,
                role="assistant",
                text="keep assistant",
                occurred_at="2026-07-12T12:01:00+00:00",
            ),
            SelectedTranscriptMessage(
                message_id="msg_new_u",
                host_message_id="host_new_u",
                generation_id="generation_2",
                source_namespace="openclaw:selected",
                source_seq=3,
                role="user",
                text="selected user",
                occurred_at="2026-07-12T12:19:00+00:00",
            ),
            SelectedTranscriptMessage(
                message_id="msg_new_a",
                host_message_id="host_new_a",
                generation_id="generation_2",
                source_namespace="openclaw:selected",
                source_seq=4,
                role="assistant",
                text=new_assistant_text,
                occurred_at="2026-07-12T12:20:00+00:00",
            ),
        ],
    )


async def _seed_derived_state(runtime: AppRuntime) -> None:
    connection = await runtime.open_connection()
    try:
        memories = MemoryObjectRepository(connection, runtime.clock)
        await memories.create_memory_object(
            memory_id="mem_abandoned",
            user_id=USER_ID,
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CHAT,
            canonical_text="derived from abandoned turn",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            payload={"source_message_ids": ["msg_old_u"]},
            extraction_hash="hash_abandoned",
            conversation_id=CONVERSATION_ID,
        )
        await memories.create_memory_object(
            memory_id="mem_edited",
            user_id=USER_ID,
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CHAT,
            canonical_text="user edited memory",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            payload={"source_message_ids": ["msg_old_a", "msg_keep_a"]},
            extraction_hash="hash_edited",
            conversation_id=CONVERSATION_ID,
        )
        await connection.execute(
            """
            INSERT INTO memory_edit_history(
                memory_id, previous_text, new_text, edited_by, edit_source, created_at
            ) VALUES (?, ?, ?, 'user', 'api', ?)
            """,
            (
                "mem_edited",
                "original",
                "user edited memory",
                runtime.clock.now().isoformat(),
            ),
        )

        payload = b"abandoned attachment"
        payload_id = "apb_abandoned"
        await ArtifactPayloadRepository(
            connection,
            runtime.clock,
        ).create_payload_blob(
            payload_blob_id=payload_id,
            user_id=USER_ID,
            storage_kind="sqlite_blob",
            identity_kind="content_sha256",
            content_sha256=hashlib.sha256(payload).hexdigest(),
            byte_size=len(payload),
            blob_bytes=payload,
            storage_key=None,
            external_uri=None,
        )
        await ArtifactRepository(connection, runtime.clock).create_artifact(
            artifact_id="art_abandoned",
            user_id=USER_ID,
            workspace_id=None,
            conversation_id=CONVERSATION_ID,
            message_id="msg_old_a",
            artifact_type="file",
            source_kind="upload",
            payload_blob_id=payload_id,
        )
        graph = EntityGraphRepository(connection, runtime.clock)
        abandoned_entity = await graph.create_entity(
            entity_id="ent_abandoned",
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            assistant_mode_id="general_qa",
            entity_type="person",
            display_name="Abandoned person",
            commit=False,
        )
        retained_entity = await graph.create_entity(
            entity_id="ent_retained",
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            assistant_mode_id="general_qa",
            entity_type="person",
            display_name="Retained person",
            commit=False,
        )
        projection = await graph.create_projection_run(
            run_id="gpr_abandoned",
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            source_message_id="msg_old_u",
            commit=False,
        )
        await graph.upsert_mention(
            user_id=USER_ID,
            entity_id=str(abandoned_entity["id"]),
            source_kind="message",
            source_id="msg_old_u",
            surface_text="Abandoned person",
            evidence_quote="abandoned user",
            conversation_id=CONVERSATION_ID,
            message_id="msg_old_u",
            projection_run_id=str(projection["id"]),
            commit=False,
        )
        await graph.upsert_mention(
            user_id=USER_ID,
            entity_id=str(retained_entity["id"]),
            source_kind="message",
            source_id="msg_keep_u",
            surface_text="Retained person",
            evidence_quote="keep user",
            conversation_id=CONVERSATION_ID,
            message_id="msg_keep_u",
            commit=False,
        )
        timestamp = runtime.clock.now().isoformat()
        await connection.execute(
            """
            INSERT INTO proxy_turn_runs(
                pair_id, request_message_id, response_message_id,
                user_id, conversation_id, request_message_role,
                request_source_seq, response_source_seq, state,
                client_fingerprint_version, client_request_fingerprint,
                final_fingerprint_version, final_provider_fingerprint,
                owner_fence, response_linked_at, durable_job_ids_json,
                completed_at, created_at, updated_at
            ) VALUES (
                'pair_abandoned', 'msg_old_u', 'msg_old_a', ?, ?, 'user',
                3, 4, 'completed', 1, 'client-fingerprint',
                1, 'provider-fingerprint', 1, ?, '[]', ?, ?, ?
            )
            """,
            (
                USER_ID,
                CONVERSATION_ID,
                timestamp,
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        for message_id, pair_role, message_role, counterpart in (
            ("msg_old_u", "request", "user", "msg_old_a"),
            ("msg_old_a", "response", "assistant", "msg_old_u"),
        ):
            await connection.execute(
                """
                INSERT INTO proxy_message_id_claims(
                    message_id, pair_id, pair_role, message_role,
                    counterpart_message_id, user_id, conversation_id,
                    claim_token, created_at
                ) VALUES (?, 'pair_abandoned', ?, ?, ?, ?, ?, 'test-claim', ?)
                """,
                (
                    message_id,
                    pair_role,
                    message_role,
                    counterpart,
                    USER_ID,
                    CONVERSATION_ID,
                    timestamp,
                ),
            )
        await connection.commit()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_replacement_is_atomic_idempotent_and_cleans_abandoned_state(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        await _seed_derived_state(runtime)
        service = SelectedTranscriptService(runtime)

        response = await service.replace(
            conversation_id=CONVERSATION_ID,
            request=_request(),
        )

        assert response.status == "rebuilding"
        assert response.stage == "preparing"
        assert response.abandoned_message_count == 2
        assert response.poll_path == (
            f"/v1/conversations/{encode_path_id(CONVERSATION_ID)}"
            f"/selected-transcript/{encode_path_id('host/op?1')}"
        )

        connection = await runtime.open_connection()
        try:
            cursor = await connection.execute(
                "SELECT id, text FROM messages WHERE conversation_id = ? ORDER BY seq",
                (CONVERSATION_ID,),
            )
            assert [(row["id"], row["text"]) for row in await cursor.fetchall()] == [
                ("msg_keep_u", "keep user"),
                ("msg_keep_a", "keep assistant"),
                ("msg_new_u", "selected user"),
                ("msg_new_a", "selected assistant"),
            ]
            cursor = await connection.execute(
                """
                SELECT
                    json_extract(metadata_json, '$.ingest_origin') AS ingest_origin,
                    json_extract(
                        metadata_json,
                        '$.confirmation_strategy'
                    ) AS confirmation_strategy
                FROM messages
                WHERE id = 'msg_new_u'
                """
            )
            inserted_metadata = await cursor.fetchone()
            assert inserted_metadata is not None
            assert inserted_metadata["ingest_origin"] == "backfill"
            assert inserted_metadata["confirmation_strategy"] == "admin_review_only"
            cursor = await connection.execute(
                """
                SELECT
                    json_extract(
                        recovery_envelope_json,
                        '$.payload.ingest_origin'
                    ) AS ingest_origin,
                    json_extract(
                        recovery_envelope_json,
                        '$.payload.confirmation_strategy'
                    ) AS confirmation_strategy
                FROM worker_job_runs
                WHERE transcript_rebuild_id = ?
                  AND job_type = 'extract_memory_candidates'
                  AND EXISTS (
                      SELECT 1
                      FROM json_each(source_message_ids_json) AS source
                      WHERE CAST(source.value AS TEXT) = 'msg_new_u'
                  )
                """,
                (response.workflow_id,),
            )
            extraction_job = await cursor.fetchone()
            assert extraction_job is not None
            assert extraction_job["ingest_origin"] == "backfill"
            assert extraction_job["confirmation_strategy"] == "admin_review_only"
            cursor = await connection.execute(
                "SELECT id, payload_json FROM memory_objects WHERE id IN (?, ?) ORDER BY id",
                ("mem_abandoned", "mem_edited"),
            )
            rows = await cursor.fetchall()
            assert [row["id"] for row in rows] == ["mem_edited"]
            assert "msg_old_a" not in str(rows[0]["payload_json"])
            assert "msg_keep_a" in str(rows[0]["payload_json"])
            for table, row_id in (
                ("artifacts", "art_abandoned"),
                ("artifact_payload_blobs", "apb_abandoned"),
            ):
                cursor = await connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE id = ?",
                    (row_id,),
                )
                assert int((await cursor.fetchone())["count"]) == 0
            for table, row_id in (
                ("graph_entities", "ent_abandoned"),
                ("graph_projection_runs", "gpr_abandoned"),
                ("proxy_turn_runs", "pair_abandoned"),
            ):
                cursor = await connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE id = ?"
                    if table != "proxy_turn_runs"
                    else "SELECT COUNT(*) AS count FROM proxy_turn_runs WHERE pair_id = ?",
                    (row_id,),
                )
                assert int((await cursor.fetchone())["count"]) == 0
            cursor = await connection.execute(
                "SELECT COUNT(*) AS count FROM graph_entities WHERE id = 'ent_retained'"
            )
            assert int((await cursor.fetchone())["count"]) == 1
            cursor = await connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM proxy_message_id_claims
                WHERE pair_id = 'pair_abandoned'
                """
            )
            assert int((await cursor.fetchone())["count"]) == 0
            cursor = await connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM worker_job_runs
                WHERE transcript_rebuild_id = ?
                  AND derivation_revision = 1
                """,
                (response.workflow_id,),
            )
            assert int((await cursor.fetchone())["count"]) >= 5

            with pytest.raises(
                aiosqlite.IntegrityError,
                match="selected transcript rebuild blocks user message writes",
            ):
                await MessageRepository(connection, runtime.clock).create_message(
                    "msg_illegal",
                    CONVERSATION_ID,
                    "user",
                    5,
                    "must be blocked",
                )
        finally:
            await connection.close()

        replay = await service.replace(
            conversation_id=CONVERSATION_ID,
            request=_request(),
        )
        assert replay.workflow_id == response.workflow_id
        assert replay.idempotent_replay is True

        with pytest.raises(
            TranscriptSelectionConflictError,
            match="operation_id already belongs to a different transcript selection",
        ):
            await service.replace(
                conversation_id=CONVERSATION_ID,
                request=_request(new_assistant_text="different selected assistant"),
            )
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_field",
    ["host_message_id", "generation_id", "source_namespace"],
)
async def test_retained_message_requires_exact_stable_host_identity(
    tmp_path: Path,
    identity_field: str,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        canonical_request = _request()
        canonical_identity = {
            "host_message_id": canonical_request.messages[0].host_message_id,
            "generation_id": canonical_request.messages[0].generation_id,
            "source_namespace": canonical_request.messages[0].source_namespace,
        }
        connection = await runtime.open_connection()
        try:
            await connection.execute(
                """
                UPDATE messages
                SET metadata_json = ?
                WHERE id = 'msg_keep_u'
                """,
                (
                    json_utils.dumps(
                        {
                            "legacy_marker": "preserve",
                            "selected_transcript_identity": canonical_identity,
                        },
                        sort_keys=True,
                    ),
                ),
            )
            await connection.commit()
        finally:
            await connection.close()

        changed_message = canonical_request.messages[0].model_copy(
            update={identity_field: f"changed-{identity_field}"}
        )
        changed_request = canonical_request.model_copy(
            update={
                "operation_id": f"identity-conflict-{identity_field}",
                "messages": [changed_message, *canonical_request.messages[1:]],
            }
        )
        with pytest.raises(
            TranscriptSelectionConflictError,
            match="different host provenance",
        ):
            await SelectedTranscriptService(runtime).replace(
                conversation_id=CONVERSATION_ID,
                request=changed_request,
            )

        connection = await runtime.open_connection()
        try:
            row = await (
                await connection.execute(
                    "SELECT metadata_json FROM messages WHERE id = 'msg_keep_u'"
                )
            ).fetchone()
            workflow_count = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) AS count FROM transcript_rebuild_workflows"
                        )
                    ).fetchone()
                )["count"]
            )
        finally:
            await connection.close()
        assert row is not None
        assert json_utils.loads(str(row["metadata_json"])) == {
            "legacy_marker": "preserve",
            "selected_transcript_identity": canonical_identity,
        }
        assert workflow_count == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_retained_legacy_identity_backfill_preserves_object_metadata(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        connection = await runtime.open_connection()
        try:
            await connection.execute(
                """
                UPDATE messages
                SET metadata_json = '{"legacy_marker":"preserve"}'
                WHERE id = 'msg_keep_u'
                """
            )
            await connection.commit()
        finally:
            await connection.close()

        request = _request(operation_id="legacy-identity-backfill")
        await SelectedTranscriptService(runtime).replace(
            conversation_id=CONVERSATION_ID,
            request=request,
        )
        connection = await runtime.open_connection()
        try:
            row = await (
                await connection.execute(
                    "SELECT metadata_json FROM messages WHERE id = 'msg_keep_u'"
                )
            ).fetchone()
        finally:
            await connection.close()
        assert row is not None
        assert json_utils.loads(str(row["metadata_json"])) == {
            "legacy_marker": "preserve",
            "selected_transcript_identity": {
                "host_message_id": request.messages[0].host_message_id,
                "generation_id": request.messages[0].generation_id,
                "source_namespace": request.messages[0].source_namespace,
            },
        }
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_retained_non_object_metadata_fails_without_mutation(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        connection = await runtime.open_connection()
        try:
            await connection.execute(
                """
                UPDATE messages
                SET metadata_json = '["legacy-causal-data"]'
                WHERE id = 'msg_keep_u'
                """
            )
            await connection.commit()
        finally:
            await connection.close()

        with pytest.raises(
            TranscriptSelectionConflictError,
            match="non-object metadata",
        ):
            await SelectedTranscriptService(runtime).replace(
                conversation_id=CONVERSATION_ID,
                request=_request(operation_id="non-object-metadata"),
            )
        connection = await runtime.open_connection()
        try:
            row = await (
                await connection.execute(
                    "SELECT metadata_json FROM messages WHERE id = 'msg_keep_u'"
                )
            ).fetchone()
            workflow_count = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) AS count FROM transcript_rebuild_workflows"
                        )
                    ).fetchone()
                )["count"]
            )
        finally:
            await connection.close()
        assert row is not None and row["metadata_json"] == '["legacy-causal-data"]'
        assert workflow_count == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_selected_messages_and_jobs_use_role_aware_source_presence(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        response = await SelectedTranscriptService(runtime).replace(
            conversation_id=CONVERSATION_ID,
            request=_request(operation_id="selected-presence-attribution"),
        )
        connection = await runtime.open_connection()
        try:
            rows = await (
                await connection.execute(
                    """
                    SELECT id, active_presence_id, source_presence_id
                    FROM messages
                    WHERE id IN ('msg_new_u', 'msg_new_a')
                    ORDER BY seq
                    """
                )
            ).fetchall()
            job_rows = await (
                await connection.execute(
                    """
                    SELECT
                        json_extract(
                            recovery_envelope_json,
                            '$.payload.message_id'
                        ) AS message_id,
                        json_extract(
                            recovery_envelope_json,
                            '$.payload.source_presence_id'
                        ) AS source_presence_id
                    FROM worker_job_runs
                    WHERE transcript_rebuild_id = ?
                      AND json_extract(
                          recovery_envelope_json,
                          '$.payload.message_id'
                      ) IN ('msg_new_u', 'msg_new_a')
                    """,
                    (response.workflow_id,),
                )
            ).fetchall()
        finally:
            await connection.close()
        assert [
            (row["id"], row["active_presence_id"], row["source_presence_id"])
            for row in rows
        ] == [
            ("msg_new_u", "default_assistant", "human_owner"),
            ("msg_new_a", "default_assistant", "default_assistant"),
        ]
        job_sources: dict[str, set[str]] = {}
        for row in job_rows:
            job_sources.setdefault(str(row["message_id"]), set()).add(
                str(row["source_presence_id"])
            )
        assert job_sources == {
            "msg_new_u": {"human_owner"},
            "msg_new_a": {"default_assistant"},
        }
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retained_source_presence_id", "expected_source_presence_id"),
    [("assistant_old", "assistant_old"), (None, None)],
)
async def test_retained_assistant_preserves_historical_presence_provenance(
    tmp_path: Path,
    retained_source_presence_id: str | None,
    expected_source_presence_id: str | None,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        connection = await runtime.open_connection()
        try:
            await connection.execute(
                """
                UPDATE conversations
                SET active_presence_id = 'assistant_new'
                WHERE id = ? AND user_id = ?
                """,
                (CONVERSATION_ID, USER_ID),
            )
            await connection.execute(
                """
                UPDATE messages
                SET active_presence_id = NULL, source_presence_id = ?
                WHERE id = 'msg_keep_a'
                """,
                (retained_source_presence_id,),
            )
            await connection.commit()
        finally:
            await connection.close()

        await SelectedTranscriptService(runtime).replace(
            conversation_id=CONVERSATION_ID,
            request=_request(
                operation_id=(
                    "retained-presence-known"
                    if retained_source_presence_id is not None
                    else "retained-presence-unknown"
                )
            ),
        )

        connection = await runtime.open_connection()
        try:
            row = await (
                await connection.execute(
                    """
                    SELECT active_presence_id, source_presence_id
                    FROM messages
                    WHERE id = 'msg_keep_a'
                    """
                )
            ).fetchone()
        finally:
            await connection.close()
        assert row is not None
        assert row["active_presence_id"] is None
        assert row["source_presence_id"] == expected_source_presence_id
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_proxy_pair_evidence_survives_until_last_selected_message_is_deleted(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        await _seed_derived_state(runtime)
        base = _request(operation_id="retain-half-proxy-pair")
        retained_request = SelectedTranscriptMessage(
            message_id="msg_old_u",
            host_message_id="host_old_u",
            generation_id="generation_old",
            source_namespace="openclaw:selected",
            source_seq=3,
            role="user",
            text="abandoned user",
            occurred_at="2026-07-12T12:09:00+00:00",
        )
        replacement_response = base.messages[3].model_copy(
            update={
                "message_id": "msg_replacement_a",
                "host_message_id": "host_replacement_a",
            }
        )
        request = base.model_copy(
            update={
                "retained_cutoff_message_id": "msg_old_u",
                "messages": [
                    base.messages[0],
                    base.messages[1],
                    retained_request,
                    replacement_response,
                ],
            }
        )
        await SelectedTranscriptService(runtime).replace(
            conversation_id=CONVERSATION_ID,
            request=request,
        )

        connection = await runtime.open_connection()
        try:
            run = await (
                await connection.execute(
                    "SELECT state FROM proxy_turn_runs WHERE pair_id = 'pair_abandoned'"
                )
            ).fetchone()
            claim_count = int(
                (
                    await (
                        await connection.execute(
                            """
                            SELECT COUNT(*) AS count
                            FROM proxy_message_id_claims
                            WHERE pair_id = 'pair_abandoned'
                            """
                        )
                    ).fetchone()
                )["count"]
            )
            assert run is not None and run["state"] == "completed"
            assert claim_count == 2
            with pytest.raises(MessageIdConflictError):
                await MessageRepository(connection, runtime.clock).create_message(
                    "msg_old_a",
                    CONVERSATION_ID,
                    "assistant",
                    8,
                    "must remain claimed",
                )
            await connection.execute(
                """
                UPDATE conversation_transcript_selections
                SET state = 'complete'
                WHERE user_id = ? AND conversation_id = ?
                """,
                (USER_ID, CONVERSATION_ID),
            )
            await connection.commit()
            await connection.execute("DELETE FROM messages WHERE id = 'msg_old_u'")
            await connection.commit()
            run_count = int(
                (
                    await (
                        await connection.execute(
                            """
                            SELECT COUNT(*) AS count
                            FROM proxy_turn_runs
                            WHERE pair_id = 'pair_abandoned'
                            """
                        )
                    ).fetchone()
                )["count"]
            )
            remaining_claims = int(
                (
                    await (
                        await connection.execute(
                            """
                            SELECT COUNT(*) AS count
                            FROM proxy_message_id_claims
                            WHERE pair_id = 'pair_abandoned'
                            """
                        )
                    ).fetchone()
                )["count"]
            )
        finally:
            await connection.close()
        assert (run_count, remaining_claims) == (0, 0)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_stale_orchestrator_cannot_restore_remediation_after_retry(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        service = SelectedTranscriptService(runtime)
        response = await service.replace(
            conversation_id=CONVERSATION_ID,
            request=_request(operation_id="stale-remediation-after-retry"),
        )

        connection = await runtime.open_connection()
        blocker = await runtime.open_connection()
        try:
            fence_clock = FrozenClock(runtime.clock.now())
            cursor = await connection.execute(
                """
                SELECT *
                FROM worker_job_runs
                WHERE job_id = (
                    SELECT orchestrator_job_id
                    FROM transcript_rebuild_workflows
                    WHERE id = ?
                )
                """,
                (response.workflow_id,),
            )
            old_job = await cursor.fetchone()
            assert old_job is not None
            await connection.execute(
                """
                UPDATE worker_job_runs
                SET status = 'running',
                    execution_owner = 'stale-worker',
                    execution_fence = execution_fence + 1,
                    execution_lease_expires_at = ?
                WHERE job_id = ?
                """,
                (
                    (fence_clock.now() + timedelta(minutes=5)).isoformat(),
                    old_job["job_id"],
                ),
            )
            await connection.commit()
            cursor = await connection.execute(
                "SELECT * FROM worker_job_runs WHERE job_id = ?",
                (old_job["job_id"],),
            )
            claimed_row = await cursor.fetchone()
            assert claimed_row is not None
            old_claim = ClaimedJob(
                notification_message_id="notification-stale-worker",
                envelope=JobEnvelope.model_validate(
                    json_utils.loads(str(claimed_row["recovery_envelope_json"]))
                ),
                owner_id="stale-worker",
                attempt_count=int(claimed_row["attempt_count"]),
                execution_fence=int(claimed_row["execution_fence"]),
                lifecycle_epoch=str(claimed_row["lifecycle_epoch"]),
                lifecycle_cleanup_key=str(claimed_row["lifecycle_cleanup_key"]),
                derivation_revision=int(claimed_row["derivation_revision"]),
            )
            repository = TranscriptRebuildRepository(connection, fence_clock)
            await connection.execute(
                """
                UPDATE worker_job_runs
                SET execution_lease_expires_at = ?
                WHERE job_id = ?
                """,
                (
                    (fence_clock.now() + timedelta(seconds=1)).isoformat(),
                    old_claim.envelope.job_id,
                ),
            )
            await connection.commit()
            await blocker.execute("BEGIN IMMEDIATE")
            observed_connection = _ObserveImmediate(connection)
            blocked_transition = asyncio.create_task(
                TranscriptRebuildRepository(
                    observed_connection,
                    fence_clock,
                ).transition_stage(
                    old_claim,
                    expected_stage="preparing",
                    next_stage="sources",
                )
            )
            await asyncio.wait_for(
                observed_connection.immediate_attempted.wait(),
                timeout=2.0,
            )
            fence_clock.advance(seconds=2)
            await blocker.commit()
            assert not await asyncio.wait_for(blocked_transition, timeout=2.0)

            await connection.execute(
                """
                UPDATE worker_job_runs
                SET execution_lease_expires_at = ?
                WHERE job_id = ?
                """,
                (
                    (fence_clock.now() + timedelta(seconds=1)).isoformat(),
                    old_claim.envelope.job_id,
                ),
            )
            await connection.commit()
            await blocker.execute("BEGIN IMMEDIATE")
            observed_connection = _ObserveImmediate(connection)
            blocked_remediation = asyncio.create_task(
                TranscriptRebuildRepository(
                    observed_connection,
                    fence_clock,
                ).mark_remediation_required(
                    old_claim,
                    expected_stage="preparing",
                    error_code="blocked_until_expired",
                    error_message="must not commit after the lease expires",
                )
            )
            await asyncio.wait_for(
                observed_connection.immediate_attempted.wait(),
                timeout=2.0,
            )
            fence_clock.advance(seconds=2)
            await blocker.commit()
            assert not await asyncio.wait_for(blocked_remediation, timeout=2.0)

            await connection.execute(
                """
                UPDATE worker_job_runs
                SET execution_lease_expires_at = ?
                WHERE job_id = ?
                """,
                (
                    (fence_clock.now() + timedelta(seconds=1)).isoformat(),
                    old_claim.envelope.job_id,
                ),
            )
            await connection.commit()
            workflow = await repository.get_workflow(response.workflow_id)
            assert workflow is not None
            await blocker.execute("BEGIN IMMEDIATE")
            observed_connection = _ObserveImmediate(connection)
            blocked_checkpoint = asyncio.create_task(
                TranscriptRebuildWorker(
                    storage_backend=runtime.storage_backend,
                    connection=observed_connection,
                    clock=fence_clock,
                    settings=runtime.settings,
                    embedding_index=runtime.embedding_index,
                )._complete_preparation(old_claim, workflow)
            )
            await asyncio.wait_for(
                observed_connection.immediate_attempted.wait(),
                timeout=2.0,
            )
            fence_clock.advance(seconds=2)
            await blocker.commit()
            with pytest.raises(
                JobLeaseLostError,
                match="preparation checkpoint lost its fence",
            ):
                await asyncio.wait_for(blocked_checkpoint, timeout=2.0)

            await connection.execute(
                """
                UPDATE worker_job_runs
                SET execution_lease_expires_at = ?
                WHERE job_id = ?
                """,
                (
                    (fence_clock.now() - timedelta(minutes=5)).isoformat(),
                    old_claim.envelope.job_id,
                ),
            )
            await connection.commit()
            assert not await repository.mark_remediation_required(
                old_claim,
                expected_stage="preparing",
                error_code="expired_failure",
                error_message="expired leases cannot mutate workflows",
            )
            assert not await repository.transition_stage(
                old_claim,
                expected_stage="preparing",
                next_stage="sources",
            )
            workflow = await repository.get_workflow(response.workflow_id)
            assert workflow is not None
            with pytest.raises(
                JobLeaseLostError,
                match="preparation checkpoint lost its fence",
            ):
                await TranscriptRebuildWorker(
                    storage_backend=runtime.storage_backend,
                    connection=connection,
                    clock=fence_clock,
                    settings=runtime.settings,
                    embedding_index=runtime.embedding_index,
                )._complete_preparation(old_claim, workflow)
            workflow = await repository.get_workflow(response.workflow_id)
            assert workflow is not None
            assert workflow["stage"] == "preparing"
            assert workflow["embedding_cleanup_completed_at"] is None
            assert workflow["error_code"] is None
            await connection.execute(
                """
                UPDATE worker_job_runs
                SET execution_lease_expires_at = ?
                WHERE job_id = ?
                """,
                (
                    (fence_clock.now() + timedelta(minutes=5)).isoformat(),
                    old_claim.envelope.job_id,
                ),
            )
            await connection.commit()
            assert await repository.mark_remediation_required(
                old_claim,
                expected_stage="preparing",
                error_code="initial_failure",
                error_message="initial current owner failure",
            )
        finally:
            if blocker.in_transaction:
                await blocker.rollback()
            await blocker.close()
            await connection.close()

        retried = await service.retry(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            operation_id="stale-remediation-after-retry",
        )
        assert retried.stage == "preparing"

        inspection = await runtime.open_connection()
        try:
            repository = TranscriptRebuildRepository(inspection, runtime.clock)
            assert not await repository.mark_remediation_required(
                old_claim,
                expected_stage="preparing",
                error_code="stale_failure",
                error_message="must not overwrite the retry",
            )
            workflow = await repository.get_workflow(response.workflow_id)
            assert workflow is not None
            assert workflow["stage"] == "preparing"
            assert workflow["orchestrator_job_id"] != old_claim.envelope.job_id
            assert workflow["error_code"] is None
            assert workflow["error_message"] is None
            selection = await repository.get_current_selection(
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
            )
            assert selection is not None
            assert selection["state"] == "rebuilding"
            cursor = await inspection.execute(
                "SELECT status FROM worker_job_runs WHERE job_id = ?",
                (old_claim.envelope.job_id,),
            )
            stale_job = await cursor.fetchone()
            assert stale_job is not None
            assert stale_job["status"] == "cancelled"
        finally:
            await inspection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_undo_rebuilds_activity_and_removes_abandoned_retrieval_trace(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        connection = await runtime.open_connection()
        try:
            await RetrievalEventRepository(connection, runtime.clock).create_event(
                {
                    "id": "ret_old_branch",
                    "user_id": USER_ID,
                    "conversation_id": CONVERSATION_ID,
                    "request_message_id": "msg_old_u",
                    "response_message_id": "msg_old_a",
                    "assistant_mode_id": "general_qa",
                    "platform_id": PLATFORM_ID,
                    "retrieval_plan_json": {"query": "abandoned"},
                    "selected_memory_ids_json": [],
                    "context_view_json": {"branch": "old"},
                    "outcome_json": {"used": True},
                    "created_at": "2026-07-12T12:09:30+00:00",
                }
            )
            stats = await ConversationActivityService(
                runtime
            ).refresh_conversation_activity_stats(
                connection,
                USER_ID,
                CONVERSATION_ID,
                as_of="2026-07-12T12:11:00+00:00",
            )
            assert stats is not None
            assert stats["message_count"] == 4
            assert stats["retrieval_count"] == 1
        finally:
            await connection.close()

        request = _request(operation_id="undo/activity", selection_epoch=1)
        request.mutation_kind = "undo"
        request.messages = request.messages[:2]
        await SelectedTranscriptService(runtime).replace(
            conversation_id=CONVERSATION_ID,
            request=request,
        )

        connection = await runtime.open_connection()
        try:
            cursor = await connection.execute(
                """
                SELECT
                    conversation.last_activity_at,
                    activity.last_message_at,
                    activity.message_count,
                    activity.user_message_count,
                    activity.assistant_message_count,
                    activity.retrieval_count
                FROM conversations AS conversation
                JOIN conversation_activity_stats AS activity
                  ON activity.user_id = conversation.user_id
                 AND activity.conversation_id = conversation.id
                WHERE conversation.id = ? AND conversation.user_id = ?
                """,
                (CONVERSATION_ID, USER_ID),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row["last_activity_at"] == "2026-07-12T12:01:00+00:00"
            assert row["last_message_at"] == "2026-07-12T12:01:00+00:00"
            assert row["message_count"] == 2
            assert row["user_message_count"] == 1
            assert row["assistant_message_count"] == 1
            assert row["retrieval_count"] == 0
            cursor = await connection.execute(
                "SELECT COUNT(*) AS count FROM retrieval_events WHERE id = ?",
                ("ret_old_branch",),
            )
            assert int((await cursor.fetchone())["count"]) == 0
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_undo_invalidates_zero_source_chunk_and_forces_recompaction(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        connection = await runtime.open_connection()
        try:
            timestamp = runtime.clock.now().isoformat()
            await SummaryRepository(connection, runtime.clock).create_summary(
                USER_ID,
                {
                    "id": "sum_old_branch",
                    "conversation_id": CONVERSATION_ID,
                    "source_message_start_seq": 1,
                    "source_message_end_seq": 4,
                    "summary_kind": SummaryViewKind.CONVERSATION_CHUNK,
                    "hierarchy_level": 0,
                    "summary_text": "summary revealing the abandoned branch",
                    "source_object_ids_json": [],
                    "maya_score": 1.0,
                    "model": "test-model",
                    "created_at": timestamp,
                },
            )
            await MemoryObjectRepository(
                connection,
                runtime.clock,
            ).upsert_summary_mirror(
                user_id=USER_ID,
                summary_view_id="sum_old_branch",
                summary_kind=SummaryViewKind.CONVERSATION_CHUNK,
                hierarchy_level=0,
                summary_text="summary revealing the abandoned branch",
                source_object_ids=[],
                created_at=timestamp,
                scope=MemoryScope.CHAT,
                conversation_id=CONVERSATION_ID,
                assistant_mode_id="general_qa",
                payload={"source_message_ids": ["msg_old_u", "msg_old_a"]},
            )
        finally:
            await connection.close()

        request = _request(operation_id="undo/chunk", selection_epoch=1)
        request.mutation_kind = "undo"
        request.messages = request.messages[:2]
        response = await SelectedTranscriptService(runtime).replace(
            conversation_id=CONVERSATION_ID,
            request=request,
        )

        connection = await runtime.open_connection()
        try:
            cursor = await connection.execute(
                "SELECT COUNT(*) AS count FROM summary_views WHERE id = ?",
                ("sum_old_branch",),
            )
            assert int((await cursor.fetchone())["count"]) == 0
            cursor = await connection.execute(
                "SELECT COUNT(*) AS count FROM memory_objects WHERE id = ?",
                (summary_mirror_id("sum_old_branch"),),
            )
            assert int((await cursor.fetchone())["count"]) == 0
            cursor = await connection.execute(
                """
                SELECT recovery_envelope_json
                FROM worker_job_runs
                WHERE transcript_rebuild_id = ?
                  AND job_type = 'compact_summaries'
                """,
                (response.workflow_id,),
            )
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert '"force_rebuild":true' in str(rows[0]["recovery_envelope_json"])
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_undo_fail_closes_profile_derived_from_abandoned_message(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        connection = await runtime.open_connection()
        try:
            timestamp = runtime.clock.now().isoformat()
            await connection.execute(
                """
                INSERT INTO user_communication_profiles(
                    id, user_id, profile_kind, scope_canonical,
                    assistant_mode_id, platform_id, profile_json,
                    source_refs_json, status, stale, created_at, updated_at
                ) VALUES (?, ?, 'user_language_profile', 'user', ?, ?, ?, ?,
                          'active', 0, ?, ?)
                """,
                (
                    "ucp_abandoned",
                    USER_ID,
                    "general_qa",
                    PLATFORM_ID,
                    '{"explicit_language_preferences":[{"language_code":"fr",'
                    '"preference_kind":"default","confidence":1.0,'
                    '"source_refs":[{"source_kind":"source_message",'
                    f'"conversation_id":"{CONVERSATION_ID}",'
                    '"source_message_id":"msg_old_u"}]}]}',
                    '[{"source_kind":"source_message",'
                    f'"conversation_id":"{CONVERSATION_ID}",'
                    '"source_message_id":"msg_old_u"}]',
                    timestamp,
                    timestamp,
                ),
            )
            await connection.commit()
        finally:
            await connection.close()

        request = _request(operation_id="undo/profile", selection_epoch=1)
        request.mutation_kind = "undo"
        request.messages = request.messages[:2]
        await SelectedTranscriptService(runtime).replace(
            conversation_id=CONVERSATION_ID,
            request=request,
        )

        connection = await runtime.open_connection()
        try:
            cursor = await connection.execute(
                """
                SELECT stale, stale_reason
                FROM user_communication_profiles
                WHERE id = ?
                """,
                ("ucp_abandoned",),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row["stale"] == 1
            assert row["stale_reason"] == "selected_transcript_rebuild"
            cursor = await connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM transcript_rebuild_targets
                WHERE user_id = ? AND message_id = 'msg_keep_u'
                """,
                (USER_ID,),
            )
            assert int((await cursor.fetchone())["count"]) == 1
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_replacement_preserves_artifact_with_a_retained_link(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        connection = await runtime.open_connection()
        try:
            payload = b"shared attachment"
            await ArtifactPayloadRepository(
                connection,
                runtime.clock,
            ).create_payload_blob(
                payload_blob_id="apb_shared",
                user_id=USER_ID,
                storage_kind="sqlite_blob",
                identity_kind="content_sha256",
                content_sha256=hashlib.sha256(payload).hexdigest(),
                byte_size=len(payload),
                blob_bytes=payload,
                storage_key=None,
                external_uri=None,
            )
            artifacts = ArtifactRepository(connection, runtime.clock)
            await artifacts.create_artifact(
                artifact_id="art_shared",
                user_id=USER_ID,
                workspace_id=None,
                conversation_id=CONVERSATION_ID,
                message_id="msg_old_a",
                artifact_type="file",
                source_kind="upload",
                payload_blob_id="apb_shared",
            )
            await artifacts.create_artifact_link(
                user_id=USER_ID,
                message_id="msg_keep_a",
                artifact_id="art_shared",
            )
        finally:
            await connection.close()

        await SelectedTranscriptService(runtime).replace(
            conversation_id=CONVERSATION_ID,
            request=_request(),
        )

        connection = await runtime.open_connection()
        try:
            cursor = await connection.execute(
                "SELECT message_id, payload_blob_id FROM artifacts WHERE id = ?",
                ("art_shared",),
            )
            artifact = await cursor.fetchone()
            assert artifact is not None
            assert artifact["message_id"] is None
            assert artifact["payload_blob_id"] == "apb_shared"
            cursor = await connection.execute(
                "SELECT COUNT(*) AS count FROM artifact_payload_blobs WHERE id = ?",
                ("apb_shared",),
            )
            assert int((await cursor.fetchone())["count"]) == 1
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_historical_selected_message_cannot_create_user_confirmation(
    tmp_path: Path,
) -> None:
    runtime = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime)
        request = _request(operation_id="historical/high-risk", selection_epoch=1)
        request.messages[2].text = "My banking card PIN is 4512."
        response = await SelectedTranscriptService(runtime).replace(
            conversation_id=CONVERSATION_ID,
            request=request,
        )

        connection = await runtime.open_connection()
        try:
            cursor = await connection.execute(
                """
                SELECT recovery_envelope_json
                FROM worker_job_runs
                WHERE transcript_rebuild_id = ?
                  AND job_type = 'extract_memory_candidates'
                  AND EXISTS (
                      SELECT 1
                      FROM json_each(source_message_ids_json) AS source
                      WHERE CAST(source.value AS TEXT) = 'msg_new_u'
                  )
                """,
                (response.workflow_id,),
            )
            job = await cursor.fetchone()
            assert job is not None
            envelope = job["recovery_envelope_json"]
            if isinstance(envelope, str):
                envelope = json_utils.loads(envelope)
            payload = envelope["payload"]
            assert payload["ingest_origin"] == "backfill"
            assert payload["confirmation_strategy"] == "admin_review_only"

            worker = IngestWorker(
                storage_backend=runtime.storage_backend,
                connection=connection,
                llm_client=runtime.llm_client,
                clock=runtime.clock,
                manifest_loader=runtime.manifest_loader,
                embedding_index=runtime.embedding_index,
                settings=runtime.settings,
            )
            context = ExtractionConversationContext(
                user_id=USER_ID,
                conversation_id=CONVERSATION_ID,
                source_message_id="msg_new_u",
                workspace_id=None,
                assistant_mode_id="general_qa",
                platform_id=PLATFORM_ID,
                ingest_origin=payload["ingest_origin"],
                confirmation_strategy=payload["confirmation_strategy"],
            )
            policy = runtime.policy_resolver.resolve(
                runtime.manifests["general_qa"],
                None,
                None,
            )
            batch = await worker._extractor._persist_result(
                result=ExtractionResult(
                    evidences=[
                        ExtractedEvidence(
                            canonical_text="My banking card PIN is 4512.",
                            index_text="banking card PIN",
                            scope=MemoryScope.USER,
                            confidence=0.99,
                            source_kind=MemorySourceKind.EXTRACTED,
                            source_quote="My banking card PIN is 4512.",
                            privacy_level=3,
                            sensitivity=MemorySensitivity.SECRET,
                            memory_category=MemoryCategory.PIN_OR_PASSWORD,
                            preserve_verbatim=True,
                            language_codes=["en"],
                        )
                    ]
                ),
                message_text="My banking card PIN is 4512.",
                role="user",
                context=context,
                resolved_policy=policy,
                cold_start=True,
                explicit_user_statement=True,
            )
            assert len(batch.persisted) == 1
            assert batch.persisted[0]["status"] == MemoryStatus.REVIEW_REQUIRED.value
            assert batch.persisted[0]["payload_json"]["ingest_origin"] == "backfill"
            assert (
                batch.persisted[0]["payload_json"]["confirmation_strategy"]
                == "admin_review_only"
            )
            cursor = await connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM pending_memory_confirmations
                WHERE user_id = ?
                """,
                (USER_ID,),
            )
            assert int((await cursor.fetchone())["count"]) == 0
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cross_runtime_same_operation_serializes_to_one_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime_a)
        service_a = SelectedTranscriptService(runtime_a)
        service_b = SelectedTranscriptService(runtime_b)
        inside_write = asyncio.Event()
        release_write = asyncio.Event()
        original_replace = service_a._replace_message_suffix

        async def paused_replace(*args: object, **kwargs: object) -> None:
            await original_replace(*args, **kwargs)
            inside_write.set()
            await release_write.wait()

        monkeypatch.setattr(service_a, "_replace_message_suffix", paused_replace)
        request = _request(operation_id="cross-runtime/same", selection_epoch=1)
        first_task = asyncio.create_task(
            service_a.replace(
                conversation_id=CONVERSATION_ID,
                request=request,
            )
        )
        await asyncio.wait_for(inside_write.wait(), timeout=2.0)
        second_task = asyncio.create_task(
            service_b.replace(
                conversation_id=CONVERSATION_ID,
                request=request.model_copy(deep=True),
            )
        )
        await asyncio.sleep(0.05)
        assert not second_task.done()
        release_write.set()
        first, second = await asyncio.gather(first_task, second_task)

        assert first.workflow_id == second.workflow_id
        assert first.idempotent_replay is False
        assert second.idempotent_replay is True
        connection = await runtime_a.open_connection()
        try:
            cursor = await connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM transcript_rebuild_workflows
                WHERE user_id = ? AND conversation_id = ?
                """,
                (USER_ID, CONVERSATION_ID),
            )
            assert int((await cursor.fetchone())["count"]) == 1
        finally:
            await connection.close()
    finally:
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_cross_runtime_rejects_conflicting_or_second_conversation_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    try:
        await _seed_conversation(runtime_a)
        setup = await runtime_a.open_connection()
        try:
            await ConversationRepository(
                setup,
                runtime_a.clock,
            ).create_conversation(
                "cnv_second",
                USER_ID,
                None,
                "general_qa",
                "Second branch",
                platform_id=PLATFORM_ID,
            )
            messages = MessageRepository(setup, runtime_a.clock)
            for message_id, role, seq, text in (
                ("second_keep_u", "user", 1, "second keep user"),
                ("second_keep_a", "assistant", 2, "second keep assistant"),
                ("second_old_u", "user", 3, "second old user"),
                ("second_old_a", "assistant", 4, "second old assistant"),
            ):
                await messages.create_message(
                    message_id,
                    "cnv_second",
                    role,
                    seq,
                    text,
                )
        finally:
            await setup.close()

        service_a = SelectedTranscriptService(runtime_a)
        inside_write = asyncio.Event()
        release_write = asyncio.Event()
        original_replace = service_a._replace_message_suffix

        async def paused_replace(*args: object, **kwargs: object) -> None:
            await original_replace(*args, **kwargs)
            inside_write.set()
            await release_write.wait()

        monkeypatch.setattr(service_a, "_replace_message_suffix", paused_replace)
        first_task = asyncio.create_task(
            service_a.replace(
                conversation_id=CONVERSATION_ID,
                request=_request(operation_id="cross-runtime/first"),
            )
        )
        await asyncio.wait_for(inside_write.wait(), timeout=2.0)

        conflicting = _request(operation_id="cross-runtime/conflict")
        conflicting.messages[-1].text = "conflicting selected assistant"
        conflict_task = asyncio.create_task(
            SelectedTranscriptService(runtime_b).replace(
                conversation_id=CONVERSATION_ID,
                request=conflicting,
            )
        )
        second_request = ReplaceSelectedTranscriptRequest(
            user_id=USER_ID,
            platform_id=PLATFORM_ID,
            operation_id="cross-runtime/second-conversation",
            selection_epoch=1,
            mutation_kind="undo",
            retained_cutoff_message_id="second_keep_a",
            messages=[
                SelectedTranscriptMessage(
                    message_id=message_id,
                    host_message_id=f"host:{message_id}",
                    generation_id="second:generation",
                    source_namespace="openclaw:selected",
                    source_seq=seq,
                    role=role,
                    text=text,
                )
                for message_id, role, seq, text in (
                    ("second_keep_u", "user", 1, "second keep user"),
                    ("second_keep_a", "assistant", 2, "second keep assistant"),
                )
            ],
        )
        second_conversation_task = asyncio.create_task(
            SelectedTranscriptService(runtime_b).replace(
                conversation_id="cnv_second",
                request=second_request,
            )
        )
        await asyncio.sleep(0.05)
        assert not conflict_task.done()
        assert not second_conversation_task.done()
        release_write.set()
        await first_task
        with pytest.raises(TranscriptSelectionConflictError):
            await conflict_task
        with pytest.raises(TranscriptRebuildInProgressError):
            await second_conversation_task

        connection = await runtime_a.open_connection()
        try:
            cursor = await connection.execute(
                """
                SELECT conversation_id, state
                FROM conversation_transcript_selections
                WHERE user_id = ?
                """,
                (USER_ID,),
            )
            rows = await cursor.fetchall()
            assert [(row["conversation_id"], row["state"]) for row in rows] == [
                (CONVERSATION_ID, "rebuilding")
            ]
        finally:
            await connection.close()
    finally:
        await runtime_b.close()
        await runtime_a.close()


def _admin_rebuild_service(
    runtime: AppRuntime,
    connection: aiosqlite.Connection,
) -> AdminRebuildService:
    return AdminRebuildService(
        connection=connection,
        llm_client=runtime.llm_client,
        embedding_index=runtime.embedding_index,
        clock=runtime.clock,
        manifest_loader=runtime.manifest_loader,
        settings=runtime.settings,
        storage_backend=runtime.storage_backend,
        job_connection_factory=runtime.open_connection,
    )


@pytest.mark.asyncio
async def test_selected_transcript_winner_blocks_admin_before_purge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    connection = await runtime_a.open_connection()
    try:
        await _seed_conversation(runtime_a)
        await SelectedTranscriptService(runtime_b).replace(
            conversation_id=CONVERSATION_ID,
            request=_request(operation_id="selected-wins-before-admin"),
        )

        service = _admin_rebuild_service(runtime_a, connection)
        purge_called = False
        original_purge = service._purge_conversation_state

        async def tracked_purge(user_id: str, conversation_id: str) -> None:
            nonlocal purge_called
            purge_called = True
            await original_purge(user_id, conversation_id)

        monkeypatch.setattr(service, "_purge_conversation_state", tracked_purge)
        with pytest.raises(TranscriptRebuildInProgressError):
            await service.rebuild_conversation(USER_ID, CONVERSATION_ID)
        assert purge_called is False
    finally:
        await connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_admin_rebuild_fence_blocks_selected_transcript_cross_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    connection = await runtime_a.open_connection()
    release_rebuild = asyncio.Event()
    rebuild_started = asyncio.Event()
    try:
        await _seed_conversation(runtime_a)
        service = _admin_rebuild_service(runtime_a, connection)

        async def paused_rebuild(*args: object, **kwargs: object) -> None:
            del args, kwargs
            rebuild_started.set()
            await asyncio.wait_for(release_rebuild.wait(), timeout=5.0)

        monkeypatch.setattr(service, "_rebuild_conversations", paused_rebuild)
        rebuild_task = asyncio.create_task(
            service.rebuild_conversation(USER_ID, CONVERSATION_ID)
        )
        await asyncio.wait_for(rebuild_started.wait(), timeout=5.0)

        with pytest.raises(TranscriptRebuildInProgressError):
            await SelectedTranscriptService(runtime_b).replace(
                conversation_id=CONVERSATION_ID,
                request=_request(operation_id="admin-wins-before-selected"),
            )

        release_rebuild.set()
        await rebuild_task
        inspection = await runtime_b.open_connection()
        try:
            cursor = await inspection.execute(
                """
                SELECT status
                FROM admin_maintenance_operations
                WHERE operation_kind = 'rebuild_conversation'
                """
            )
            assert [row["status"] for row in await cursor.fetchall()] == ["succeeded"]
            cursor = await inspection.execute(
                "SELECT COUNT(*) AS count FROM transcript_rebuild_workflows"
            )
            assert int((await cursor.fetchone())["count"]) == 0
        finally:
            await inspection.close()
    finally:
        release_rebuild.set()
        await connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_failed_admin_maintenance_releases_scope_durably(
    tmp_path: Path,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    try:
        await _seed_conversation(runtime_a)
        failed_operation = None
        with pytest.raises(RuntimeError, match="injected admin failure"):
            async with admin_maintenance_operation(
                connection_a,
                runtime_a.clock,
                operation_kind="failure_probe",
                user_id=USER_ID,
            ) as operation:
                failed_operation = operation
                await JobRunRepository(
                    connection_a,
                    runtime_a.clock,
                ).create_durable_job(
                    stream_name="atagia:evaluate",
                    target_backend="inprocess",
                    envelope=JobEnvelope(
                        job_id="job_failed_admin_operation",
                        job_type=JobType.RUN_EVALUATION,
                        user_id=USER_ID,
                        maintenance_operation_id=operation.operation_id,
                        payload={"metrics": ["system"]},
                    ),
                    source_token_estimate=None,
                    size_bucket=None,
                )
                raise RuntimeError("injected admin failure")

        cursor = await connection_b.execute(
            """
            SELECT status, error_class
            FROM admin_maintenance_operations
            WHERE operation_kind = 'failure_probe'
            """
        )
        failed = await cursor.fetchone()
        assert failed is not None
        assert (failed["status"], failed["error_class"]) == ("failed", "RuntimeError")
        cursor = await connection_b.execute(
            """
            SELECT status, recovery_envelope_json
            FROM worker_job_runs
            WHERE job_id = 'job_failed_admin_operation'
            """
        )
        failed_job = await cursor.fetchone()
        assert failed_job is not None
        assert failed_job["status"] == "cancelled"
        assert failed_job["recovery_envelope_json"] is None
        assert failed_operation is not None
        await AdminMaintenanceRepository(
            connection_a,
            runtime_a.clock,
        ).fail(failed_operation, RuntimeError("repeated release"))

        async with admin_maintenance_operation(
            connection_b,
            runtime_b.clock,
            operation_kind="recovery_probe",
            user_id=USER_ID,
        ) as recovered_operation:
            pass
        await AdminMaintenanceRepository(
            connection_b,
            runtime_b.clock,
        ).succeed(recovered_operation)
    finally:
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_global_and_user_maintenance_acquisition_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    try:
        await _seed_conversation(runtime_a)
        async with admin_maintenance_operation(
            connection_a,
            runtime_a.clock,
            operation_kind="user_scope_probe",
            user_id=USER_ID,
        ):
            with pytest.raises(TranscriptRebuildInProgressError):
                await AdminMaintenanceRepository(
                    connection_b,
                    runtime_b.clock,
                ).acquire_global(operation_kind="blocked_global_probe")

        async with admin_maintenance_operation(
            connection_a,
            runtime_a.clock,
            operation_kind="global_scope_probe",
            user_id=None,
        ):
            with pytest.raises(TranscriptRebuildInProgressError):
                await AdminMaintenanceRepository(
                    connection_b,
                    runtime_b.clock,
                ).acquire_user(
                    user_id=USER_ID,
                    operation_kind="blocked_user_probe",
                )
            with pytest.raises(TranscriptRebuildInProgressError):
                await SelectedTranscriptService(runtime_b).replace(
                    conversation_id=CONVERSATION_ID,
                    request=_request(operation_id="blocked-by-global-maintenance"),
                )
    finally:
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_expired_crashed_maintenance_owner_no_longer_blocks_selected_transcript(
    tmp_path: Path,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    crashed_connection = await runtime_a.open_connection()
    try:
        await _seed_conversation(runtime_a)
        crashed = await AdminMaintenanceRepository(
            crashed_connection,
            runtime_a.clock,
        ).acquire_user(
            user_id=USER_ID,
            operation_kind="crashed_owner_probe",
            lease_seconds=0.1,
        )
        await asyncio.sleep(0.25)

        response = await SelectedTranscriptService(runtime_b).replace(
            conversation_id=CONVERSATION_ID,
            request=_request(operation_id="after-crashed-admin-owner"),
        )
        assert response.status == "rebuilding"
        with pytest.raises(TranscriptRebuildInProgressError):
            await AdminMaintenanceRepository(
                crashed_connection,
                runtime_a.clock,
            ).require_current(crashed)
    finally:
        await crashed_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_admin_heartbeat_waiting_past_expiry_cannot_revive_owner(
    tmp_path: Path,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    owner_connection = await runtime_a.open_connection()
    blocker_connection = await runtime_b.open_connection()
    try:
        await _seed_conversation(runtime_a)
        repository = AdminMaintenanceRepository(
            owner_connection,
            runtime_a.clock,
        )
        operation = await repository.acquire_user(
            user_id=USER_ID,
            operation_kind="heartbeat_lock_wait_probe",
            lease_seconds=0.1,
        )
        cursor = await owner_connection.execute(
            """
            SELECT lease_expires_at
            FROM admin_maintenance_operations
            WHERE id = ?
            """,
            (operation.operation_id,),
        )
        original_expiry = str((await cursor.fetchone())["lease_expires_at"])

        await blocker_connection.execute("BEGIN IMMEDIATE")
        heartbeat_task = asyncio.create_task(repository.heartbeat(operation))
        await asyncio.sleep(0.2)
        assert not heartbeat_task.done()
        await blocker_connection.commit()

        assert await heartbeat_task is False
        cursor = await owner_connection.execute(
            """
            SELECT heartbeat_at, lease_expires_at
            FROM admin_maintenance_operations
            WHERE id = ?
            """,
            (operation.operation_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert str(row["lease_expires_at"]) == original_expiry
    finally:
        if blocker_connection.in_transaction:
            await blocker_connection.rollback()
        await blocker_connection.close()
        await owner_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_expired_dirty_maintenance_requires_matching_remediation_retry(
    tmp_path: Path,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    crashed_connection = await runtime_a.open_connection()
    retry_connection = await runtime_b.open_connection()
    try:
        await _seed_conversation(runtime_a)
        repository = AdminMaintenanceRepository(
            crashed_connection,
            runtime_a.clock,
        )
        crashed = await repository.acquire_user(
            user_id=USER_ID,
            operation_kind="dirty_crash_probe",
            recovery_key="conversation:selected",
            lease_seconds=0.1,
        )
        await crashed_connection.execute("BEGIN IMMEDIATE")
        await repository.require_current(crashed)
        await repository.mark_dirty(crashed)
        await crashed_connection.commit()
        await asyncio.sleep(0.25)

        with pytest.raises(TranscriptRebuildInProgressError):
            await SelectedTranscriptService(runtime_b).replace(
                conversation_id=CONVERSATION_ID,
                request=_request(operation_id="blocked-by-dirty-crash"),
            )
        with pytest.raises(TranscriptRebuildInProgressError):
            await AdminMaintenanceRepository(
                retry_connection,
                runtime_b.clock,
            ).acquire_user(
                user_id=USER_ID,
                operation_kind="different_operation",
                recovery_key="conversation:selected",
            )

        async with admin_maintenance_operation(
            retry_connection,
            runtime_b.clock,
            operation_kind="dirty_crash_probe",
            user_id=USER_ID,
            recovery_key="conversation:selected",
        ) as resumed:
            assert resumed.operation_id == crashed.operation_id
            assert resumed.phase == "dirty"
            assert resumed.resumed is True

        cursor = await crashed_connection.execute(
            "SELECT status FROM admin_maintenance_operations WHERE id = ?",
            (crashed.operation_id,),
        )
        assert (await cursor.fetchone())["status"] == "succeeded"
        response = await SelectedTranscriptService(runtime_b).replace(
            conversation_id=CONVERSATION_ID,
            request=_request(operation_id="after-dirty-remediation"),
        )
        assert response.status == "rebuilding"
    finally:
        await retry_connection.close()
        await crashed_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_next_admin_acquisition_recovers_expired_owner_and_jobs(
    tmp_path: Path,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    try:
        await _seed_conversation(runtime_a)
        expired = await AdminMaintenanceRepository(
            connection_a,
            runtime_a.clock,
        ).acquire_user(
            user_id=USER_ID,
            operation_kind="expired_owner_probe",
            lease_seconds=0.1,
        )
        await JobRunRepository(
            connection_a,
            runtime_a.clock,
        ).create_durable_job(
            stream_name="atagia:evaluate",
            target_backend="inprocess",
            envelope=JobEnvelope(
                job_id="job_expired_admin_operation",
                job_type=JobType.RUN_EVALUATION,
                user_id=USER_ID,
                maintenance_operation_id=expired.operation_id,
                payload={"metrics": ["system"]},
            ),
            source_token_estimate=None,
            size_bucket=None,
        )
        await asyncio.sleep(0.25)

        async with admin_maintenance_operation(
            connection_b,
            runtime_b.clock,
            operation_kind="replacement_owner_probe",
            user_id=USER_ID,
        ):
            pass

        cursor = await connection_a.execute(
            """
            SELECT status, error_class
            FROM admin_maintenance_operations
            WHERE id = ?
            """,
            (expired.operation_id,),
        )
        recovered = await cursor.fetchone()
        assert recovered is not None
        assert (recovered["status"], recovered["error_class"]) == (
            "failed",
            "AdminMaintenanceLeaseExpired",
        )
        cursor = await connection_a.execute(
            """
            SELECT status, error_class
            FROM worker_job_runs
            WHERE job_id = 'job_expired_admin_operation'
            """
        )
        recovered_job = await cursor.fetchone()
        assert recovered_job is not None
        assert (recovered_job["status"], recovered_job["error_class"]) == (
            "cancelled",
            "AdminMaintenanceLeaseExpired",
        )
    finally:
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_dedicated_heartbeat_keeps_long_admin_operation_owned(
    tmp_path: Path,
) -> None:
    runtime_a = await _runtime(tmp_path)
    runtime_b = await _runtime(tmp_path)
    connection = await runtime_a.open_connection()
    try:
        await _seed_conversation(runtime_a)
        async with admin_maintenance_operation(
            connection,
            runtime_a.clock,
            operation_kind="heartbeat_probe",
            user_id=USER_ID,
            heartbeat_connection_factory=runtime_b.open_connection,
            lease_seconds=0.3,
        ) as operation:
            await asyncio.sleep(0.7)
            await AdminMaintenanceRepository(
                connection,
                runtime_a.clock,
            ).require_current(operation)
    finally:
        await connection.close()
        await runtime_b.close()
        await runtime_a.close()
