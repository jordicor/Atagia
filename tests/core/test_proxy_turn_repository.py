"""Durable pair ownership, global claims, fencing, and atomic completion tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from shutil import copy2
from time import perf_counter
from typing import Any, Coroutine

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.conversation_namespace import (
    capture_conversation_namespace_snapshot,
)
from atagia.core.db_sqlite import (
    MigrationManager,
    close_connection,
    initialize_database,
    open_connection,
)
from atagia.core.proxy_turn_repository import (
    ProxyDurableJobInsert,
    ProxyTurnRepository,
    ProxyTurnRequestMessage,
    ProxyTurnResponseMessage,
    ProxyTurnTelemetry,
    proxy_turn_pair_id,
)
from atagia.core.presence_repository import PresenceRepository
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
)
from atagia.core.space_repository import SpaceRepository
from atagia.models.schemas_jobs import EXTRACT_STREAM_NAME, JobEnvelope, JobType
from atagia.models.schemas_memory import (
    ConversationStatus,
    SpaceBoundaryMode,
    TurnSurface,
)
from atagia.services.errors import (
    ConversationNotFoundError,
    MessageIdConflictError,
    ProxyTurnConflictError,
    ProxyTurnInProgressError,
    ProxyTurnStaleOwnerError,
    SourceSequenceConflictError,
)
from atagia.services.proxy_transcript import (
    ProxyInputProjection,
    bind_input_metadata,
    build_response_metadata,
)

from tests.turn_telemetry_support import sample_turn_telemetry


MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)


def _proxy_turn_telemetry(
    *,
    user_id: str,
    conversation_id: str,
    request_message_id: str,
) -> ProxyTurnTelemetry:
    """Minimal telemetry payload for a finalize() call under test."""
    telemetry = sample_turn_telemetry(TurnSurface.PROXY_COMPLETION)
    return ProxyTurnTelemetry(
        telemetry=telemetry,
        # The fenced writer re-measures the turn from this origin, so it has to
        # be a real perf_counter reading that already contains the sample's
        # retrieval slice -- exactly as in production, where retrieval runs
        # inside the turn rather than before it starts.
        turn_started_at=perf_counter() - (telemetry.turn_to_event_write_wall_ms / 1000.0),
        retrieval_event_id=None,
        fallback_event={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "request_message_id": request_message_id,
            "assistant_mode_id": None,
            "retrieval_plan_json": {},
            "selected_memory_ids_json": [],
            "context_view_json": {},
            "outcome_json": {"memory_context_available": False},
        },
    )


async def _seed(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
    *,
    user_id: str = "usr_proxy",
    conversation_id: str = "cnv_proxy",
) -> None:
    await UserRepository(connection, clock).create_user(user_id)
    await connection.execute(
        """
        INSERT OR IGNORE INTO assistant_modes(
            id, display_name, prompt_hash, memory_policy_json, created_at, updated_at
        ) VALUES ('coding_debug', 'Coding Debug', 'hash', '{}', ?, ?)
        """,
        (clock.now().isoformat(), clock.now().isoformat()),
    )
    await connection.commit()
    await ConversationRepository(connection, clock).create_conversation(
        conversation_id,
        user_id,
        None,
        "coding_debug",
        "Proxy test",
        platform_id="proxy-tests",
    )
    active_presence = await PresenceRepository(
        connection,
        clock,
    ).resolve_active_presence(
        owner_user_id=user_id,
        active_presence_id=None,
        character_id=None,
    )
    await ConversationRepository(connection, clock).set_active_presence(
        conversation_id,
        user_id,
        str(active_presence["id"]),
    )


async def _database(
    database_path: str = ":memory:",
) -> tuple[aiosqlite.Connection, FrozenClock]:
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    await _seed(connection, clock)
    return connection, clock


def _request(
    *,
    request_id: str = "msg_request",
    response_id: str = "msg_response",
    user_id: str = "usr_proxy",
    conversation_id: str = "cnv_proxy",
    role: str = "user",
    text: str = "Remember this turn.",
    source_seq: int | None = None,
    response_source_seq: int | None = None,
    fingerprint: str = "client-fingerprint",
) -> ProxyTurnRequestMessage:
    pair_id = proxy_turn_pair_id(
        user_id=user_id,
        conversation_id=conversation_id,
        request_message_id=request_id,
        response_message_id=response_id,
    )
    projection = ProxyInputProjection(
        message_role=role,
        text=text,
        tool_projection={"schema_version": 1, "kind": f"{role}_input"},
    )
    return ProxyTurnRequestMessage(
        message_id=request_id,
        response_message_id=response_id,
        user_id=user_id,
        conversation_id=conversation_id,
        role=role,  # type: ignore[arg-type]
        text=text,
        source_seq=source_seq,
        response_source_seq=response_source_seq,
        metadata=bind_input_metadata(
            projection,
            pair_id=pair_id,
            request_message_id=request_id,
            response_message_id=response_id,
            client_request_fingerprint=fingerprint,
            parent_response_message_id=("msg_parent" if role == "tool" else None),
        ),
    )


def _job(claim: object) -> ProxyDurableJobInsert:
    envelope = JobEnvelope(
        job_id="job_proxy_pair_extract",
        job_type=JobType.EXTRACT_MEMORY_CANDIDATES,
        user_id=getattr(claim, "user_id"),
        conversation_id=getattr(claim, "conversation_id"),
        message_ids=[getattr(claim, "response_message_id")],
        payload={
            "message_id": getattr(claim, "response_message_id"),
            "message_text": "Stored answer",
        },
    )
    return ProxyDurableJobInsert(
        stream_name=EXTRACT_STREAM_NAME,
        target_backend="inprocess",
        envelope=envelope,
        source_token_estimate=3,
        size_bucket="small",
        metadata={"message_count": 1},
        user_persona_id=None,
        platform_id="proxy-tests",
        character_id=None,
        incognito_snapshot=False,
        remember_across_chats_snapshot=True,
        remember_across_devices_snapshot=True,
        temporary_snapshot=False,
        purge_on_close_snapshot=False,
        policy_snapshot={"platform_id": "proxy-tests"},
    )


def _response(claim: object) -> ProxyTurnResponseMessage:
    return ProxyTurnResponseMessage(
        text="Stored answer",
        metadata=build_response_metadata(
            pair_id=getattr(claim, "pair_id"),
            request_message_id=getattr(claim, "request_message_id"),
            response_message_id=getattr(claim, "response_message_id"),
            client_request_fingerprint=getattr(claim, "client_request_fingerprint"),
            final_provider_fingerprint=getattr(claim, "final_provider_fingerprint"),
            content="Stored answer",
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            model="proxy-model",
        ),
        source_seq=getattr(claim, "response_source_seq", None),
    )


class _ObserveImmediate:
    """Expose the point where a connection starts waiting for a writer lock."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self.immediate_attempted = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if " ".join(sql.split()).upper() == "BEGIN IMMEDIATE":
            self.immediate_attempted.set()
        return await self._connection.execute(sql, *args, **kwargs)


async def _run_after_writer_waits(
    *,
    blocker: aiosqlite.Connection,
    observed: _ObserveImmediate,
    clock: FrozenClock,
    operation: Coroutine[Any, Any, Any],
) -> Any:
    await blocker.execute("BEGIN IMMEDIATE")
    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(observed.immediate_attempted.wait(), timeout=2.0)
        clock.advance(seconds=2)
        await blocker.commit()
        return await asyncio.wait_for(task, timeout=2.0)
    finally:
        if blocker.in_transaction:
            await blocker.rollback()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _set_active_space(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
    space_id: str,
) -> None:
    await SpaceRepository(connection, clock).resolve_space(
        owner_user_id="usr_proxy",
        space_id=space_id,
        boundary_mode=SpaceBoundaryMode.FOCUS,
        display_name=space_id,
        source_kind="explicit",
        source_id=space_id,
    )
    await ConversationRepository(connection, clock).set_active_space(
        "cnv_proxy",
        "usr_proxy",
        space_id,
    )


@pytest.mark.asyncio
async def test_reservation_claims_both_ids_and_fences_takeover() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        request = _request()
        first = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
            lease_seconds=10,
        )
        assert first.created is True
        assert first.claim is not None
        claims = await repository.get_claims(first.claim.pair_id)
        assert {
            (claim["message_id"], claim["pair_role"], claim["message_role"])
            for claim in claims
        } == {
            ("msg_request", "request", "user"),
            ("msg_response", "response", "assistant"),
        }
        with pytest.raises(ProxyTurnInProgressError) as live:
            await repository.reserve(
                request_message=request,
                client_request_fingerprint="client-fingerprint",
                lease_seconds=10,
            )
        assert live.value.code == "request_in_progress"

        clock.advance(seconds=11)
        takeover = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
            lease_seconds=10,
        )
        assert takeover.takeover is True
        assert takeover.claim is not None
        assert takeover.claim.owner_fence == first.claim.owner_fence + 1
        with pytest.raises(ProxyTurnStaleOwnerError):
            await repository.establish_final_fingerprint(
                first.claim,
                "final-fingerprint",
            )
        current = await repository.establish_final_fingerprint(
            takeover.claim,
            "final-fingerprint",
        )
        assert current.final_provider_fingerprint == "final-fingerprint"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_completed_pair_replays_without_recreating_jobs() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        request = _request()
        reservation = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
        )
        await repository.finalize(
            claim,
            response=_response(claim),
            durable_jobs=[_job(claim)],
            turn_telemetry=_proxy_turn_telemetry(
                user_id=claim.user_id,
                conversation_id=claim.conversation_id,
                request_message_id=claim.request_message_id,
            ),
        )

        replay = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
        )
        assert replay.replay is not None
        assert replay.replay.replay_envelope["content"] == "Stored answer"
        assert replay.replay.replay_envelope["finish_reason"] == "stop"
        cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM worker_job_runs WHERE job_id = ?",
            ("job_proxy_pair_extract",),
        )
        assert int((await cursor.fetchone())["count"]) == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failpoint_name",
    [
        "response_inserted",
        "durable_job_inserted:job_proxy_pair_extract",
        "response_linked",
        "run_completed",
    ],
)
async def test_terminal_failpoints_roll_back_response_jobs_and_completed_state(
    tmp_path: Path,
    failpoint_name: str,
) -> None:
    database_path = str(tmp_path / "terminal-failpoint.db")
    connection, clock = await _database(database_path)
    try:
        repository = ProxyTurnRepository(connection, clock)
        reservation = await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
        )

        def failpoint(name: str) -> None:
            if name == failpoint_name:
                raise RuntimeError(f"injected:{name}")

        with pytest.raises(RuntimeError, match="injected"):
            await repository.finalize(
                claim,
                response=_response(claim),
                durable_jobs=[_job(claim)],
                turn_telemetry=_proxy_turn_telemetry(
                    user_id=claim.user_id,
                    conversation_id=claim.conversation_id,
                    request_message_id=claim.request_message_id,
                ),
                failpoint=failpoint,
            )
        await close_connection(connection)
        connection = await open_connection(database_path)
        repository = ProxyTurnRepository(connection, clock)
        assert (
            await MessageRepository(connection, clock).get_message_for_idempotency(
                claim.response_message_id
            )
            is None
        )
        run = await repository.get_run(claim.pair_id)
        assert run is not None
        assert run["state"] == "generating"
        assert run["response_linked_at"] is None
        cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM worker_job_runs WHERE job_id = ?",
            ("job_proxy_pair_extract",),
        )
        assert int((await cursor.fetchone())["count"]) == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_finalize_rechecks_active_conversation_after_lifecycle_wins(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "terminal-lifecycle-race.db")
    connection, clock = await _database(database_path)
    mutation_connection: aiosqlite.Connection | None = None
    ready = asyncio.Event()
    resume = asyncio.Event()
    try:
        repository = ProxyTurnRepository(connection, clock)
        reservation = await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
        )

        async def finalize_after_barrier() -> None:
            ready.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            await repository.finalize(
                claim,
                response=_response(claim),
                durable_jobs=[_job(claim)],
                turn_telemetry=_proxy_turn_telemetry(
                    user_id=claim.user_id,
                    conversation_id=claim.conversation_id,
                    request_message_id=claim.request_message_id,
                ),
            )

        finalize_task = asyncio.create_task(finalize_after_barrier())
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        mutation_connection = await open_connection(database_path)
        await mutation_connection.execute(
            """
            UPDATE conversations
            SET status = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                ConversationStatus.PENDING_DELETION.value,
                clock.now().isoformat(),
                claim.conversation_id,
                claim.user_id,
            ),
        )
        await mutation_connection.commit()
        resume.set()

        with pytest.raises(ConversationNotFoundError):
            await finalize_task
        assert (
            await MessageRepository(connection, clock).get_message_for_idempotency(
                claim.response_message_id
            )
            is None
        )
        cursor = await connection.execute(
            "SELECT 1 FROM worker_job_runs WHERE job_id = ?",
            ("job_proxy_pair_extract",),
        )
        assert await cursor.fetchone() is None
        run = await repository.get_run(claim.pair_id)
        assert run is not None
        assert run["state"] == "generating"
    finally:
        resume.set()
        if mutation_connection is not None:
            await mutation_connection.close()
        await connection.close()


@pytest.mark.asyncio
async def test_exposed_stream_cannot_be_taken_over_after_lease_expiry() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        request = _request()
        reservation = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
            lease_seconds=5,
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
            lease_seconds=5,
        )
        claim = await repository.mark_emission_started(claim, lease_seconds=5)
        clock.advance(seconds=6)
        with pytest.raises(ProxyTurnConflictError) as ambiguous:
            await repository.reserve(
                request_message=request,
                client_request_fingerprint="client-fingerprint",
            )
        assert ambiguous.value.code == "stream_retry_requires_new_ids"
        run = await repository.get_run(claim.pair_id)
        assert run is not None
        assert run["state"] == "ambiguous_exposed"
        assert await repository.claim_is_current(claim) is False
        with pytest.raises(ProxyTurnStaleOwnerError):
            await repository.finalize(
                claim,
                response=_response(claim),
                durable_jobs=[_job(claim)],
                turn_telemetry=_proxy_turn_telemetry(
                    user_id=claim.user_id,
                    conversation_id=claim.conversation_id,
                    request_message_id=claim.request_message_id,
                ),
            )
        assert (
            await MessageRepository(connection, clock).get_message_for_idempotency(
                claim.response_message_id
            )
            is None
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_takeover_fences_stale_owner_before_emission_marker() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        request = _request()
        first = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
            lease_seconds=5,
        )
        assert first.claim is not None
        stale = await repository.establish_final_fingerprint(
            first.claim,
            "final-fingerprint",
            lease_seconds=5,
        )
        clock.advance(seconds=6)
        takeover = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
            lease_seconds=5,
        )
        assert takeover.claim is not None

        with pytest.raises(ProxyTurnStaleOwnerError):
            await repository.mark_emission_started(stale, lease_seconds=5)

        current = await repository.establish_final_fingerprint(
            takeover.claim,
            "final-fingerprint",
            lease_seconds=5,
        )
        emitted = await repository.mark_emission_started(current, lease_seconds=5)
        assert emitted.emission_started is True
        run = await repository.get_run(emitted.pair_id)
        assert run is not None
        assert run["state"] == "emission_started"
        assert run["owner_fence"] == emitted.owner_fence
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_id", "response_id"),
    [
        ("msg_request", "msg_other_response"),
        ("msg_other_request", "msg_response"),
        ("msg_response", "msg_request"),
        ("msg_request", "msg_request"),
    ],
)
async def test_shared_swapped_or_equal_ids_conflict_before_a_second_input_write(
    request_id: str,
    response_id: str,
) -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )
        with pytest.raises(ProxyTurnConflictError):
            await repository.reserve(
                request_message=_request(
                    request_id=request_id,
                    response_id=response_id,
                    fingerprint="other-fingerprint",
                ),
                client_request_fingerprint="other-fingerprint",
            )
        cursor = await connection.execute("SELECT COUNT(*) AS count FROM messages")
        assert int((await cursor.fetchone())["count"]) == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_direct_message_insert_cannot_use_a_reserved_proxy_id() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )
        await connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(MessageIdConflictError):
            await MessageRepository(connection, clock).create_message(
                message_id="msg_response",
                conversation_id="cnv_proxy",
                role="assistant",
                seq=None,
                text="Attempted direct insert",
                commit=False,
            )
        assert connection.in_transaction
        await connection.rollback()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pair_evidence_survives_pruning_until_last_message_is_deleted() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        reservation = await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
        )
        await repository.finalize(
            claim,
            response=_response(claim),
            durable_jobs=[_job(claim)],
            turn_telemetry=_proxy_turn_telemetry(
                user_id=claim.user_id,
                conversation_id=claim.conversation_id,
                request_message_id=claim.request_message_id,
            ),
        )
        clock.advance(seconds=60)
        assert await repository.prune_terminal_diagnostics(before=clock.now()) == 1
        assert (await repository.get_run(claim.pair_id))["state"] == "completed"

        await connection.execute(
            "DELETE FROM messages WHERE id = ?",
            (claim.request_message_id,),
        )
        await connection.commit()
        assert await repository.get_run(claim.pair_id) is not None
        replay = await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )
        assert replay.replay is not None
        assert replay.replay.replay_envelope["content"] == "Stored answer"
        await connection.execute(
            "DELETE FROM messages WHERE id = ?",
            (claim.response_message_id,),
        )
        await connection.commit()
        assert await repository.get_run(claim.pair_id) is None
        assert await repository.get_claims(claim.pair_id) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_request_and_response_sequences_are_durable_and_replay_validated() -> (
    None
):
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        request = _request(source_seq=5, response_source_seq=6)
        reservation = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
        )
        await repository.finalize(
            claim,
            response=_response(claim),
            durable_jobs=[_job(claim)],
            turn_telemetry=_proxy_turn_telemetry(
                user_id=claim.user_id,
                conversation_id=claim.conversation_id,
                request_message_id=claim.request_message_id,
            ),
        )
        run = await repository.get_run(claim.pair_id)
        assert run is not None
        assert (run["request_source_seq"], run["response_source_seq"]) == (5, 6)

        omitted = await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )
        assert omitted.replay is not None
        with pytest.raises(SourceSequenceConflictError):
            await repository.reserve(
                request_message=_request(source_seq=4),
                client_request_fingerprint="client-fingerprint",
            )
        with pytest.raises(SourceSequenceConflictError):
            await repository.reserve(
                request_message=_request(response_source_seq=7),
                client_request_fingerprint="client-fingerprint",
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_invalid_response_sequence_rolls_back_pair_and_input() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        with pytest.raises(SourceSequenceConflictError):
            await repository.reserve(
                request_message=_request(source_seq=5, response_source_seq=5),
                client_request_fingerprint="client-fingerprint",
            )
        cursor = await connection.execute("SELECT COUNT(*) AS count FROM messages")
        assert int((await cursor.fetchone())["count"]) == 0
        cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM proxy_turn_runs"
        )
        assert int((await cursor.fetchone())["count"]) == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_global_claims_reject_reuse_across_user_and_conversation_namespaces() -> (
    None
):
    connection, clock = await _database()
    try:
        await _seed(
            connection,
            clock,
            user_id="usr_proxy_other",
            conversation_id="cnv_proxy_other_user",
        )
        await ConversationRepository(connection, clock).create_conversation(
            "cnv_proxy_other_conversation",
            "usr_proxy",
            None,
            "coding_debug",
            "Other conversation",
            platform_id="proxy-tests",
        )
        repository = ProxyTurnRepository(connection, clock)
        await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )
        for user_id, conversation_id in (
            ("usr_proxy_other", "cnv_proxy_other_user"),
            ("usr_proxy", "cnv_proxy_other_conversation"),
        ):
            with pytest.raises(ProxyTurnConflictError) as conflict:
                await repository.reserve(
                    request_message=_request(
                        response_id=f"msg_response_{conversation_id}",
                        user_id=user_id,
                        conversation_id=conversation_id,
                        fingerprint="other-fingerprint",
                    ),
                    client_request_fingerprint="other-fingerprint",
                )
            assert conflict.value.code == "proxy_message_id_claim_conflict"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_takeover_with_changed_final_provider_request_is_permanent_conflict() -> (
    None
):
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        request = _request()
        reservation = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
            lease_seconds=5,
        )
        assert reservation.claim is not None
        await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint-a",
            lease_seconds=5,
        )
        clock.advance(seconds=6)
        takeover = await repository.reserve(
            request_message=request,
            client_request_fingerprint="client-fingerprint",
        )
        assert takeover.claim is not None
        with pytest.raises(ProxyTurnConflictError) as changed:
            await repository.establish_final_fingerprint(
                takeover.claim,
                "final-fingerprint-b",
            )
        assert changed.value.code == "proxy_final_request_changed"
        await repository.prune_terminal_diagnostics(before=clock.now())
        with pytest.raises(ProxyTurnConflictError) as permanent:
            await repository.reserve(
                request_message=request,
                client_request_fingerprint="client-fingerprint",
            )
        assert permanent.value.code == "proxy_final_request_changed"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_migration_0056_backfills_only_reciprocal_legacy_pairs(
    tmp_path: Path,
) -> None:
    bootstrap_migrations = tmp_path / "migrations-through-0055"
    bootstrap_migrations.mkdir()
    for migration in MigrationManager(MIGRATIONS_DIR).discover():
        if migration.version <= 55:
            copy2(migration.path, bootstrap_migrations / migration.path.name)
    database_path = tmp_path / "proxy-upgrade.db"
    connection = await initialize_database(str(database_path), bootstrap_migrations)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    await _seed(connection, clock)
    pair_id = proxy_turn_pair_id(
        user_id="usr_proxy",
        conversation_id="cnv_proxy",
        request_message_id="msg_legacy_request",
        response_message_id="msg_legacy_response",
    )
    projection = ProxyInputProjection(
        message_role="user",
        text="Legacy request",
        tool_projection={"schema_version": 1, "kind": "user_input"},
    )
    input_metadata = bind_input_metadata(
        projection,
        pair_id=pair_id,
        request_message_id="msg_legacy_request",
        response_message_id="msg_legacy_response",
        client_request_fingerprint="legacy-client-fingerprint",
    )
    response_metadata = build_response_metadata(
        pair_id=pair_id,
        request_message_id="msg_legacy_request",
        response_message_id="msg_legacy_response",
        client_request_fingerprint="legacy-client-fingerprint",
        final_provider_fingerprint="legacy-final-fingerprint",
        content="Legacy response",
        tool_calls=[],
        finish_reason="stop",
        usage=None,
        model="proxy-model",
    )
    timestamp = clock.now().isoformat()
    await connection.executemany(
        """
        INSERT INTO messages(
            id, conversation_id, role, seq, text, metadata_json, created_at, occurred_at
        ) VALUES (?, 'cnv_proxy', ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                "msg_legacy_request",
                "user",
                1,
                "Legacy request",
                json.dumps(input_metadata),
                timestamp,
                timestamp,
            ),
            (
                "msg_legacy_response",
                "assistant",
                2,
                "Legacy response",
                json.dumps(response_metadata),
                timestamp,
                timestamp,
            ),
            (
                "msg_legacy_unresolved",
                "user",
                3,
                "Unresolved legacy row",
                "{}",
                timestamp,
                timestamp,
            ),
        ),
    )
    await connection.commit()
    await close_connection(connection)

    upgraded = await initialize_database(str(database_path), MIGRATIONS_DIR)
    try:
        repository = ProxyTurnRepository(upgraded, clock)
        run = await repository.get_run(pair_id)
        assert run is not None
        assert run["state"] == "completed"
        assert (run["request_source_seq"], run["response_source_seq"]) == (1, 2)
        assert len(await repository.get_claims(pair_id)) == 2
        cursor = await upgraded.execute(
            "SELECT COUNT(*) AS count FROM proxy_message_id_claims WHERE message_id = ?",
            ("msg_legacy_unresolved",),
        )
        assert int((await cursor.fetchone())["count"]) == 0
        with pytest.raises(ProxyTurnConflictError):
            await repository.reserve(
                request_message=_request(
                    request_id="msg_legacy_unresolved",
                    response_id="msg_new_response",
                    text="Unresolved legacy row",
                ),
                client_request_fingerprint="client-fingerprint",
            )
        violations = await (
            await upgraded.execute("PRAGMA foreign_key_check")
        ).fetchall()
        assert violations == []
    finally:
        await close_connection(upgraded)


@pytest.mark.asyncio
async def test_migration_0063_quarantines_inflight_runs_and_snapshots_completed(
    tmp_path: Path,
) -> None:
    bootstrap_migrations = tmp_path / "migrations-through-0062"
    bootstrap_migrations.mkdir()
    migration_0063 = None
    for migration in MigrationManager(MIGRATIONS_DIR).discover():
        if migration.version <= 62:
            copy2(migration.path, bootstrap_migrations / migration.path.name)
        elif migration.version == 63:
            migration_0063 = migration
    assert migration_0063 is not None

    database_path = tmp_path / "proxy-source-snapshot-upgrade.db"
    connection = await initialize_database(str(database_path), bootstrap_migrations)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    try:
        await _seed(connection, clock)
        await connection.execute(
            """
            UPDATE user_lifecycles
            SET derivation_revision = 7
            WHERE user_id = 'usr_proxy'
            """
        )
        lifecycle = await (
            await connection.execute(
                """
                SELECT lifecycle_epoch
                FROM user_lifecycles
                WHERE user_id = 'usr_proxy'
                """
            )
        ).fetchone()
        assert lifecycle is not None
        lifecycle_epoch = str(lifecycle["lifecycle_epoch"])
        timestamp = clock.now().isoformat()
        lease_expires_at = "2026-07-13T10:30:00+00:00"
        await connection.executemany(
            """
            INSERT INTO proxy_turn_runs(
                pair_id,
                request_message_id,
                response_message_id,
                user_id,
                conversation_id,
                request_message_role,
                request_source_seq,
                response_source_seq,
                state,
                client_fingerprint_version,
                client_request_fingerprint,
                final_fingerprint_version,
                final_provider_fingerprint,
                owner_token,
                owner_fence,
                lease_expires_at,
                emission_started_at,
                response_linked_at,
                durable_job_ids_json,
                completed_at,
                created_at,
                updated_at
            ) VALUES (
                ?, ?, ?, 'usr_proxy', 'cnv_proxy', 'user', ?, ?, ?, 1, ?,
                1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                (
                    "pair_generating",
                    "msg_generating_request",
                    "msg_generating_response",
                    None,
                    None,
                    "generating",
                    "client-generating",
                    "final-generating",
                    "owner-generating",
                    3,
                    lease_expires_at,
                    None,
                    None,
                    "[]",
                    None,
                    timestamp,
                    timestamp,
                ),
                (
                    "pair_emission_started",
                    "msg_emission_request",
                    "msg_emission_response",
                    None,
                    None,
                    "emission_started",
                    "client-emission",
                    "final-emission",
                    "owner-emission",
                    4,
                    lease_expires_at,
                    timestamp,
                    None,
                    "[]",
                    None,
                    timestamp,
                    timestamp,
                ),
                (
                    "pair_completed",
                    "msg_completed_request",
                    "msg_completed_response",
                    10,
                    11,
                    "completed",
                    "client-completed",
                    "final-completed",
                    None,
                    5,
                    None,
                    None,
                    timestamp,
                    '["job_completed"]',
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            ),
        )
        await connection.commit()
    finally:
        await close_connection(connection)

    copy2(migration_0063.path, bootstrap_migrations / migration_0063.path.name)
    upgrading = await open_connection(str(database_path))
    try:
        applied = await MigrationManager(bootstrap_migrations).apply_all(upgrading)
        assert [migration.version for migration in applied] == [63]
    finally:
        await close_connection(upgrading)

    reopened = await open_connection(str(database_path))
    try:
        versions = await MigrationManager(bootstrap_migrations).applied_versions(
            reopened
        )
        assert max(versions) == 63
        rows = {
            str(row["pair_id"]): row
            for row in await (
                await reopened.execute(
                    """
                    SELECT
                        pair_id,
                        state,
                        final_provider_fingerprint,
                        owner_token,
                        owner_fence,
                        lease_expires_at,
                        emission_started_at,
                        response_linked_at,
                        durable_job_ids_json,
                        completed_at,
                        ambiguous_at,
                        error_code,
                        error_message,
                        lifecycle_epoch,
                        derivation_revision
                    FROM proxy_turn_runs
                    ORDER BY pair_id
                    """
                )
            ).fetchall()
        }

        generating = rows["pair_generating"]
        assert generating["state"] == "final_fingerprint_conflict"
        assert generating["owner_token"] is None
        assert generating["lease_expires_at"] is None
        assert generating["error_code"] == "proxy_source_snapshot_missing"
        assert (
            generating["error_message"]
            == "Pre-cutover generation has no canonical source snapshot"
        )
        assert generating["ambiguous_at"] is None
        assert generating["lifecycle_epoch"] is None
        assert generating["derivation_revision"] is None
        assert generating["final_provider_fingerprint"] == "final-generating"
        assert int(generating["owner_fence"]) == 3

        emitted = rows["pair_emission_started"]
        assert emitted["state"] == "ambiguous_exposed"
        assert emitted["owner_token"] is None
        assert emitted["lease_expires_at"] is None
        assert emitted["emission_started_at"] == timestamp
        assert emitted["ambiguous_at"] is not None
        assert emitted["error_code"] == "proxy_source_snapshot_missing"
        assert (
            emitted["error_message"]
            == "Pre-cutover stream has no canonical source snapshot"
        )
        assert emitted["lifecycle_epoch"] is None
        assert emitted["derivation_revision"] is None
        assert emitted["final_provider_fingerprint"] == "final-emission"
        assert int(emitted["owner_fence"]) == 4

        completed = rows["pair_completed"]
        assert completed["state"] == "completed"
        assert completed["owner_token"] is None
        assert completed["lease_expires_at"] is None
        assert completed["error_code"] is None
        assert completed["error_message"] is None
        assert completed["ambiguous_at"] is None
        assert completed["lifecycle_epoch"] == lifecycle_epoch
        assert int(completed["derivation_revision"]) == 7
        assert completed["response_linked_at"] == timestamp
        assert completed["durable_job_ids_json"] == '["job_completed"]'
        assert completed["completed_at"] == timestamp
        assert completed["final_provider_fingerprint"] == "final-completed"
        assert int(completed["owner_fence"]) == 5

        snapshot_rows = await (
            await reopened.execute(
                """
                SELECT pair_id
                FROM proxy_turn_runs
                WHERE lifecycle_epoch IS NOT NULL OR derivation_revision IS NOT NULL
                """
            )
        ).fetchall()
        assert [str(row["pair_id"]) for row in snapshot_rows] == ["pair_completed"]
        index = await (
            await reopened.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'index'
                  AND name = 'idx_proxy_turn_runs_user_source_snapshot'
                """
            )
        ).fetchone()
        assert index is not None
        assert (
            await (await reopened.execute("PRAGMA foreign_key_check")).fetchall() == []
        )
    finally:
        await close_connection(reopened)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direct_message_id", "direct_role"),
    (
        ("msg_request", "user"),
        ("msg_response", "assistant"),
    ),
)
async def test_reservation_race_with_direct_message_insert_has_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_message_id: str,
    direct_role: str,
) -> None:
    database_path = tmp_path / f"claim-race-{direct_role}.db"
    setup = await initialize_database(str(database_path), MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    await _seed(setup, clock)
    await close_connection(setup)
    reservation_connection = await open_connection(str(database_path))
    direct_connection = await open_connection(str(database_path))
    direct_repository = MessageRepository(direct_connection, clock)
    validation_reached = asyncio.Event()
    release_direct_insert = asyncio.Event()
    direct_validation_had_transaction: bool | None = None
    original_validate_claim = direct_repository._validate_proxy_message_id_claim

    async def pause_after_claim_validation(**kwargs) -> None:
        nonlocal direct_validation_had_transaction
        await original_validate_claim(**kwargs)
        direct_validation_had_transaction = direct_connection.in_transaction
        validation_reached.set()
        await release_direct_insert.wait()

    monkeypatch.setattr(
        direct_repository,
        "_validate_proxy_message_id_claim",
        pause_after_claim_validation,
    )

    async def reserve_pair() -> str:
        try:
            await ProxyTurnRepository(
                reservation_connection,
                clock,
            ).reserve(
                request_message=_request(),
                client_request_fingerprint="client-fingerprint",
            )
            return "reservation"
        except ProxyTurnConflictError:
            return "reservation_conflict"

    async def insert_directly() -> str:
        try:
            await direct_repository.create_message(
                message_id=direct_message_id,
                conversation_id="cnv_proxy",
                role=direct_role,
                seq=20,
                text="Direct competing insert",
            )
            return "direct"
        except MessageIdConflictError:
            return "direct_conflict"

    try:
        direct_task = asyncio.create_task(insert_directly())
        await validation_reached.wait()
        reservation_task = asyncio.create_task(reserve_pair())
        await asyncio.sleep(0)
        assert not reservation_task.done()
        release_direct_insert.set()
        results = await asyncio.gather(direct_task, reservation_task)
        assert results == ["direct", "reservation_conflict"]
        assert direct_validation_had_transaction is True
        verification = await open_connection(str(database_path))
        try:
            run_count = int(
                (
                    await (
                        await verification.execute(
                            "SELECT COUNT(*) AS count FROM proxy_turn_runs"
                        )
                    ).fetchone()
                )["count"]
            )
            claim_count = int(
                (
                    await (
                        await verification.execute(
                            "SELECT COUNT(*) AS count FROM proxy_message_id_claims"
                        )
                    ).fetchone()
                )["count"]
            )
            direct_message_count = int(
                (
                    await (
                        await verification.execute(
                            "SELECT COUNT(*) AS count FROM messages WHERE id = ?",
                            (direct_message_id,),
                        )
                    ).fetchone()
                )["count"]
            )
            assert (run_count, claim_count, direct_message_count) == (0, 0, 1)
        finally:
            await close_connection(verification)
    finally:
        await close_connection(reservation_connection)
        await close_connection(direct_connection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direct_message_id", "direct_role"),
    (
        ("msg_request", "user"),
        ("msg_response", "assistant"),
    ),
)
async def test_commit_false_without_transaction_cannot_race_proxy_reservation(
    tmp_path: Path,
    direct_message_id: str,
    direct_role: str,
) -> None:
    database_path = tmp_path / f"claim-no-transaction-{direct_role}.db"
    setup = await initialize_database(str(database_path), MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    await _seed(setup, clock)
    await close_connection(setup)
    reservation_connection = await open_connection(str(database_path))
    direct_connection = await open_connection(str(database_path))
    start = asyncio.Event()

    async def reserve_pair() -> str:
        await start.wait()
        await ProxyTurnRepository(reservation_connection, clock).reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )
        return "reservation"

    async def insert_without_transaction() -> str:
        await start.wait()
        with pytest.raises(
            RuntimeError,
            match=r"create_message\(commit=False\) requires an active caller-owned transaction",
        ):
            await MessageRepository(direct_connection, clock).create_message(
                message_id=direct_message_id,
                conversation_id="cnv_proxy",
                role=direct_role,
                seq=20,
                text="Invalid unowned transaction insert",
                commit=False,
            )
        assert not direct_connection.in_transaction
        return "usage_error"

    try:
        reservation_task = asyncio.create_task(reserve_pair())
        direct_task = asyncio.create_task(insert_without_transaction())
        start.set()
        results = await asyncio.gather(reservation_task, direct_task)
        assert results == ["reservation", "usage_error"]

        verification = await open_connection(str(database_path))
        try:
            claims = await (
                await verification.execute(
                    "SELECT message_id FROM proxy_message_id_claims ORDER BY message_id"
                )
            ).fetchall()
            messages = await (
                await verification.execute("SELECT id, role FROM messages ORDER BY id")
            ).fetchall()
            run_count = int(
                (
                    await (
                        await verification.execute(
                            "SELECT COUNT(*) AS count FROM proxy_turn_runs"
                        )
                    ).fetchone()
                )["count"]
            )
            assert [str(row["message_id"]) for row in claims] == [
                "msg_request",
                "msg_response",
            ]
            assert [(str(row["id"]), str(row["role"])) for row in messages] == [
                ("msg_request", "user")
            ]
            assert run_count == 1
        finally:
            await close_connection(verification)
    finally:
        await close_connection(reservation_connection)
        await close_connection(direct_connection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direct_message_id", "direct_role"),
    (
        ("msg_request", "user"),
        ("msg_response", "assistant"),
    ),
)
async def test_deferred_direct_insert_locks_before_proxy_claim_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_message_id: str,
    direct_role: str,
) -> None:
    database_path = tmp_path / f"claim-deferred-{direct_role}.db"
    setup = await initialize_database(str(database_path), MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    await _seed(setup, clock)
    await close_connection(setup)
    reservation_connection = await open_connection(str(database_path))
    direct_connection = await open_connection(str(database_path))
    direct_repository = MessageRepository(direct_connection, clock)
    validation_reached = asyncio.Event()
    release_direct_insert = asyncio.Event()
    validation_had_write_transaction: bool | None = None
    caller_transaction_survived_return: bool | None = None
    original_validate_claim = direct_repository._validate_proxy_message_id_claim

    async def pause_after_claim_validation(**kwargs) -> None:
        nonlocal validation_had_write_transaction
        await original_validate_claim(**kwargs)
        validation_had_write_transaction = direct_connection.in_transaction
        validation_reached.set()
        await release_direct_insert.wait()

    monkeypatch.setattr(
        direct_repository,
        "_validate_proxy_message_id_claim",
        pause_after_claim_validation,
    )

    async def reserve_pair() -> str:
        try:
            await ProxyTurnRepository(reservation_connection, clock).reserve(
                request_message=_request(),
                client_request_fingerprint="client-fingerprint",
            )
            return "reservation"
        except ProxyTurnConflictError:
            return "reservation_conflict"

    async def insert_directly() -> str:
        nonlocal caller_transaction_survived_return
        await direct_connection.execute("BEGIN")
        try:
            await direct_repository.create_message(
                message_id=direct_message_id,
                conversation_id="cnv_proxy",
                role=direct_role,
                seq=20,
                text="Deferred competing insert",
                commit=False,
            )
            caller_transaction_survived_return = direct_connection.in_transaction
            await direct_connection.commit()
            return "direct"
        except BaseException:
            await direct_connection.rollback()
            raise

    try:
        direct_task = asyncio.create_task(insert_directly())
        await validation_reached.wait()
        reservation_task = asyncio.create_task(reserve_pair())
        await asyncio.sleep(0)
        reservation_was_blocked = not reservation_task.done()
        release_direct_insert.set()
        results = await asyncio.gather(direct_task, reservation_task)
        assert reservation_was_blocked
        assert results == ["direct", "reservation_conflict"]
        assert validation_had_write_transaction is True
        assert caller_transaction_survived_return is True

        verification = await open_connection(str(database_path))
        try:
            run_count = int(
                (
                    await (
                        await verification.execute(
                            "SELECT COUNT(*) AS count FROM proxy_turn_runs"
                        )
                    ).fetchone()
                )["count"]
            )
            claim_count = int(
                (
                    await (
                        await verification.execute(
                            "SELECT COUNT(*) AS count FROM proxy_message_id_claims"
                        )
                    ).fetchone()
                )["count"]
            )
            messages = await (
                await verification.execute("SELECT id, role FROM messages ORDER BY id")
            ).fetchall()
            assert (run_count, claim_count) == (0, 0)
            assert [(str(row["id"]), str(row["role"])) for row in messages] == [
                (direct_message_id, direct_role)
            ]
        finally:
            await close_connection(verification)
    finally:
        await close_connection(reservation_connection)
        await close_connection(direct_connection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direct_message_id", "direct_role"),
    (
        ("msg_request", "user"),
        ("msg_response", "assistant"),
    ),
)
async def test_stale_deferred_snapshot_raises_stable_message_id_conflict(
    tmp_path: Path,
    direct_message_id: str,
    direct_role: str,
) -> None:
    database_path = tmp_path / f"claim-stale-deferred-{direct_role}.db"
    setup = await initialize_database(str(database_path), MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    await _seed(setup, clock)
    await close_connection(setup)
    reservation_connection = await open_connection(str(database_path))
    direct_connection = await open_connection(str(database_path))

    try:
        await direct_connection.execute("BEGIN")
        await (
            await direct_connection.execute(
                "SELECT COUNT(*) FROM proxy_message_id_claims"
            )
        ).fetchone()
        await ProxyTurnRepository(reservation_connection, clock).reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
        )

        with pytest.raises(
            MessageIdConflictError,
            match="could not serialize message_id claim validation",
        ):
            await MessageRepository(direct_connection, clock).create_message(
                message_id=direct_message_id,
                conversation_id="cnv_proxy",
                role=direct_role,
                seq=20,
                text="Stale deferred competing insert",
                commit=False,
            )
        assert direct_connection.in_transaction
        await direct_connection.rollback()

        verification = await open_connection(str(database_path))
        try:
            claims = await (
                await verification.execute(
                    "SELECT message_id FROM proxy_message_id_claims ORDER BY message_id"
                )
            ).fetchall()
            messages = await (
                await verification.execute("SELECT id, role FROM messages ORDER BY id")
            ).fetchall()
            assert [str(row["message_id"]) for row in claims] == [
                "msg_request",
                "msg_response",
            ]
            assert [(str(row["id"]), str(row["role"])) for row in messages] == [
                ("msg_request", "user")
            ]
        finally:
            await close_connection(verification)
    finally:
        if direct_connection.in_transaction:
            await direct_connection.rollback()
        await close_connection(reservation_connection)
        await close_connection(direct_connection)


@pytest.mark.asyncio
async def test_reserve_rejects_namespace_changed_after_external_validation() -> None:
    connection, clock = await _database()
    try:
        snapshot = await capture_conversation_namespace_snapshot(
            connection,
            clock,
            user_id="usr_proxy",
            conversation_id="cnv_proxy",
        )
        assert snapshot is not None
        await _set_active_space(connection, clock, "space_after_validation")

        with pytest.raises(ProxyTurnConflictError) as changed:
            await ProxyTurnRepository(connection, clock).reserve(
                request_message=_request(),
                client_request_fingerprint="client-fingerprint",
                expected_namespace_snapshot=snapshot,
            )
        assert changed.value.code == "proxy_namespace_changed"
        assert (
            await ProxyTurnRepository(
                connection,
                clock,
            ).get_run(
                proxy_turn_pair_id(
                    user_id="usr_proxy",
                    conversation_id="cnv_proxy",
                    request_message_id="msg_request",
                    response_message_id="msg_response",
                )
            )
            is None
        )
        assert (
            await MessageRepository(
                connection,
                clock,
            ).get_message_for_idempotency("msg_request")
            is None
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_namespace_change_fail_closes_unexposed_proxy_turn() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        reservation = await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
            lease_seconds=100,
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
            lease_seconds=100,
        )
        await _set_active_space(connection, clock, "space_after_provider_binding")

        with pytest.raises(ProxyTurnConflictError) as changed:
            await repository.finalize(
                claim,
                response=_response(claim),
                durable_jobs=[_job(claim)],
                turn_telemetry=_proxy_turn_telemetry(
                    user_id=claim.user_id,
                    conversation_id=claim.conversation_id,
                    request_message_id=claim.request_message_id,
                ),
            )
        assert changed.value.code == "proxy_namespace_changed"
        run = await repository.get_run(claim.pair_id)
        assert run is not None
        assert run["state"] == "final_fingerprint_conflict"
        assert run["error_code"] == "proxy_namespace_changed"
        assert (
            await MessageRepository(
                connection,
                clock,
            ).get_message_for_idempotency(claim.response_message_id)
            is None
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_namespace_change_after_emission_becomes_ambiguous_on_renew() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        reservation = await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
            lease_seconds=100,
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
            lease_seconds=100,
        )
        claim = await repository.mark_emission_started(
            claim,
            lease_seconds=100,
        )
        await _set_active_space(connection, clock, "space_after_emission")

        assert await repository.renew(claim, lease_seconds=100) is False
        run = await repository.get_run(claim.pair_id)
        assert run is not None
        assert run["state"] == "ambiguous_exposed"
        assert run["error_code"] == "proxy_namespace_changed"
        assert await repository.claim_is_current(claim) is False
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_source_revision_change_after_emission_stops_renewal() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        reservation = await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
            lease_seconds=100,
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
            lease_seconds=100,
        )
        claim = await repository.mark_emission_started(
            claim,
            lease_seconds=100,
        )
        await connection.execute(
            """
            UPDATE user_lifecycles
            SET derivation_revision = derivation_revision + 1
            WHERE user_id = ?
            """,
            (claim.user_id,),
        )
        await connection.commit()

        assert await repository.renew(claim, lease_seconds=100) is False
        run = await repository.get_run(claim.pair_id)
        assert run is not None
        assert run["state"] == "ambiguous_exposed"
        assert run["error_code"] == "proxy_source_revision_changed"
        assert await repository.claim_is_current(claim) is False
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation_name",
    ["reserve", "establish", "mark_emission", "renew"],
)
async def test_proxy_lease_deadline_is_sampled_after_writer_lock(
    tmp_path: Path,
    operation_name: str,
) -> None:
    database_path = str(tmp_path / f"proxy-lock-time-{operation_name}.db")
    setup = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    await _seed(setup, clock)
    await close_connection(setup)
    worker = await open_connection(database_path)
    blocker = await open_connection(database_path)
    try:
        base_repository = ProxyTurnRepository(worker, clock)
        claim = None
        if operation_name != "reserve":
            reservation = await base_repository.reserve(
                request_message=_request(),
                client_request_fingerprint="client-fingerprint",
                lease_seconds=100,
            )
            assert reservation.claim is not None
            claim = reservation.claim
        if operation_name in {"mark_emission", "renew"}:
            assert claim is not None
            claim = await base_repository.establish_final_fingerprint(
                claim,
                "final-fingerprint",
                lease_seconds=100,
            )
        if operation_name == "renew":
            assert claim is not None
            claim = await base_repository.mark_emission_started(
                claim,
                lease_seconds=100,
            )

        observed = _ObserveImmediate(worker)
        repository = ProxyTurnRepository(observed, clock)  # type: ignore[arg-type]
        if operation_name == "reserve":
            operation = repository.reserve(
                request_message=_request(),
                client_request_fingerprint="client-fingerprint",
                lease_seconds=1,
            )
        elif operation_name == "establish":
            assert claim is not None
            operation = repository.establish_final_fingerprint(
                claim,
                "final-fingerprint",
                lease_seconds=1,
            )
        elif operation_name == "mark_emission":
            assert claim is not None
            operation = repository.mark_emission_started(
                claim,
                lease_seconds=1,
            )
        else:
            assert claim is not None
            operation = repository.renew(claim, lease_seconds=1)

        result = await _run_after_writer_waits(
            blocker=blocker,
            observed=observed,
            clock=clock,
            operation=operation,
        )
        if operation_name == "renew":
            assert result is True
        result_claim = result.claim if operation_name == "reserve" else result
        pair_id = result_claim.pair_id if operation_name != "renew" else claim.pair_id
        run = await base_repository.get_run(pair_id)
        assert run is not None
        assert datetime.fromisoformat(run["lease_expires_at"]) > clock.now()
    finally:
        if blocker.in_transaction:
            await blocker.rollback()
        await close_connection(blocker)
        await close_connection(worker)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation_name",
    ["establish", "mark_emission", "renew"],
)
async def test_waiting_past_existing_proxy_lease_cannot_revive_owner(
    tmp_path: Path,
    operation_name: str,
) -> None:
    database_path = str(tmp_path / f"proxy-expired-wait-{operation_name}.db")
    setup = await initialize_database(database_path, MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    await _seed(setup, clock)
    await close_connection(setup)
    worker = await open_connection(database_path)
    blocker = await open_connection(database_path)
    try:
        base_repository = ProxyTurnRepository(worker, clock)
        reservation = await base_repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
            lease_seconds=1,
        )
        assert reservation.claim is not None
        claim = reservation.claim
        if operation_name in {"mark_emission", "renew"}:
            claim = await base_repository.establish_final_fingerprint(
                claim,
                "final-fingerprint",
                lease_seconds=1,
            )
        if operation_name == "renew":
            claim = await base_repository.mark_emission_started(
                claim,
                lease_seconds=1,
            )
        before = await base_repository.get_run(claim.pair_id)
        assert before is not None

        observed = _ObserveImmediate(worker)
        repository = ProxyTurnRepository(observed, clock)  # type: ignore[arg-type]
        if operation_name == "establish":
            operation = repository.establish_final_fingerprint(
                claim,
                "final-fingerprint",
                lease_seconds=100,
            )
        elif operation_name == "mark_emission":
            operation = repository.mark_emission_started(
                claim,
                lease_seconds=100,
            )
        else:
            operation = repository.renew(claim, lease_seconds=100)

        if operation_name == "renew":
            assert (
                await _run_after_writer_waits(
                    blocker=blocker,
                    observed=observed,
                    clock=clock,
                    operation=operation,
                )
                is False
            )
        else:
            with pytest.raises(ProxyTurnStaleOwnerError):
                await _run_after_writer_waits(
                    blocker=blocker,
                    observed=observed,
                    clock=clock,
                    operation=operation,
                )
        after = await base_repository.get_run(claim.pair_id)
        assert after is not None
        assert after["lease_expires_at"] == before["lease_expires_at"]
    finally:
        if blocker.in_transaction:
            await blocker.rollback()
        await close_connection(blocker)
        await close_connection(worker)


@pytest.mark.asyncio
async def test_expired_proxy_lease_cannot_finalize() -> None:
    connection, clock = await _database()
    try:
        repository = ProxyTurnRepository(connection, clock)
        reservation = await repository.reserve(
            request_message=_request(),
            client_request_fingerprint="client-fingerprint",
            lease_seconds=1,
        )
        assert reservation.claim is not None
        claim = await repository.establish_final_fingerprint(
            reservation.claim,
            "final-fingerprint",
            lease_seconds=1,
        )
        clock.advance(seconds=2)

        assert await repository.claim_is_current(claim) is False
        with pytest.raises(ProxyTurnStaleOwnerError):
            await repository.finalize(
                claim,
                response=_response(claim),
                durable_jobs=[_job(claim)],
                turn_telemetry=_proxy_turn_telemetry(
                    user_id=claim.user_id,
                    conversation_id=claim.conversation_id,
                    request_message_id=claim.request_message_id,
                ),
            )
        assert (
            await MessageRepository(
                connection,
                clock,
            ).get_message_for_idempotency(claim.response_message_id)
            is None
        )
    finally:
        await connection.close()
