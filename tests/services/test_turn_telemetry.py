"""Persisted per-turn and ingest telemetry (CS-1.5).

The engine already computed turn timings and LLM call counts, but they only
existed in the in-memory debug payload. These tests pin the durable, queryable
half: typed columns on ``retrieval_events`` for every turn surface, and the pair
of stamps on ``memory_objects`` -- source message arrival and ``queryable_at``
(migration 0071) -- that makes time-to-queryable measurable without guessing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from shutil import copy2
import sqlite3
from typing import Any

import aiosqlite
import pytest

from atagia import Atagia
from atagia.core.db_sqlite import MigrationManager, initialize_database
from atagia.core.repositories import MemoryObjectRepository, MessageRepository
from atagia.core.retrieval_event_repository import (
    RetrievalEventRepository,
    TurnTelemetry,
)
from atagia.models.schemas_memory import (
    LLMPurposeMetricsTrace,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    TurnSurface,
)

from atagia.services.context_cache_service import ContextCacheService

from tests.test_engine import EngineProvider, _install_stub_client
from tests.turn_telemetry_support import sample_turn_telemetry

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
# A database reaching the current schema either predates every telemetry column
# (68) or sits at the intermediate calls-only breakdown (69). Both paths must
# end on the same shape, and only the second exercises 0070's column swap.
_TELEMETRY_BOOTSTRAP_VERSIONS = (68, 69)


async def _engine(monkeypatch: pytest.MonkeyPatch) -> Atagia:
    provider = EngineProvider()
    _install_stub_client(monkeypatch, provider)
    engine = Atagia(
        db_path=":memory:",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    await engine.setup()
    await engine.create_user("usr_telemetry")
    await engine.create_conversation(
        "usr_telemetry", "cnv_telemetry", assistant_mode_id="coding_debug"
    )
    return engine


async def _turn_rows(
    engine: Atagia, user_id: str = "usr_telemetry"
) -> list[dict[str, Any]]:
    connection = await engine.runtime.open_connection()
    try:
        return await RetrievalEventRepository(
            connection, engine.runtime.clock
        ).list_events(user_id, None, limit=50, offset=0)
    finally:
        await connection.close()


# ---------------------------------------------------------------------------
# Chat turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_turn_persists_queryable_turn_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = await _engine(monkeypatch)
    try:
        result = await engine.chat(
            user_id="usr_telemetry",
            conversation_id="cnv_telemetry",
            mode="coding_debug",
            message="How do I fix this retry loop?",
            debug=True,
        )
        rows = await _turn_rows(engine)
    finally:
        await engine.close()

    assert result.debug is not None
    metrics = result.debug["llm_call_metrics"]
    assert len(rows) == 1
    row = rows[0]

    assert row["id"] == result.retrieval_event_id
    assert row["turn_surface"] == TurnSurface.CHAT.value
    assert row["response_message_id"] == result.response_message_id

    # Timings are real measurements, not placeholders, and the retrieval slice
    # cannot exceed the turn that contains it.
    assert row["turn_to_event_write_wall_ms"] > 0.0
    assert row["retrieval_duration_ms"] > 0.0
    assert row["retrieval_duration_ms"] <= row["turn_to_event_write_wall_ms"]
    assert row["stage_timings_ms_json"]

    # The persisted scalars are the same numbers the debug payload reports, so
    # aggregating the table cannot drift from what a single traced turn shows.
    assert row["llm_total_calls"] == metrics["total_calls"]
    assert row["llm_failed_calls"] == metrics["failed_calls"]
    assert row["llm_total_latency_ms"] == pytest.approx(metrics["total_latency_ms"])
    assert row["llm_by_purpose_json"] == metrics["by_purpose"]
    assert row["llm_by_purpose_json"]["chat_reply"]["calls"] == 1
    # CS-1.4: the same metrics are IN THE TRACE, not only in the typed columns.
    assert row["outcome_json"]["llm_call_metrics"] == metrics
    assert row["outcome_json"]["retrieval_trace"]["llm_call_metrics"] == metrics
    # CS-1.5: the column carries wall time per purpose, not only call counts.
    assert row["llm_by_purpose_json"]["chat_reply"]["latency_ms"] >= 0.0
    assert sum(
        usage["latency_ms"] for usage in row["llm_by_purpose_json"].values()
    ) == pytest.approx(row["llm_total_latency_ms"])


@pytest.mark.asyncio
async def test_chat_turn_duration_covers_the_persistence_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``turn_to_event_write_wall_ms`` is measured at the write, not at answer-ready.

    Everything after the reply -- message persistence, this very row, the
    confirmation plan, job enqueue, the commit -- is a real part of what a
    caller waits for, and it dominates a turn against a fast provider. Slowing
    one repository call down must therefore move the persisted number.
    """
    engine = await _engine(monkeypatch)
    original_create_message = MessageRepository.create_message
    delay_seconds = 0.05

    async def slow_create_message(
        self: MessageRepository, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        # Only the assistant row is delayed: it is written after the reply, so
        # a measurement that stopped at answer-ready could not see this at all.
        row = await original_create_message(self, *args, **kwargs)
        if row["role"] == "assistant":
            await asyncio.sleep(delay_seconds)
        return row

    monkeypatch.setattr(MessageRepository, "create_message", slow_create_message)
    try:
        result = await engine.chat(
            user_id="usr_telemetry",
            conversation_id="cnv_telemetry",
            mode="coding_debug",
            message="How do I fix this retry loop?",
            debug=True,
        )
        rows = await _turn_rows(engine)
    finally:
        await engine.close()

    assert result.debug is not None
    row = rows[0]
    metrics = result.debug["llm_call_metrics"]
    # The turn contains its own retrieval and its own provider calls.
    assert row["turn_to_event_write_wall_ms"] > row["retrieval_duration_ms"]
    assert row["turn_to_event_write_wall_ms"] > metrics["total_latency_ms"]
    # Regression pin: the injected post-reply delay is inside the measured span.
    assert row["turn_to_event_write_wall_ms"] > delay_seconds * 1000.0


@pytest.mark.asyncio
async def test_failed_turn_telemetry_write_fails_the_whole_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telemetry is part of the turn, not a side effect that may be skipped."""
    engine = await _engine(monkeypatch)

    async def exploding_create_event(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("telemetry write failed")

    monkeypatch.setattr(
        RetrievalEventRepository, "create_event", exploding_create_event
    )
    try:
        with pytest.raises(RuntimeError, match="telemetry write failed"):
            await engine.chat(
                user_id="usr_telemetry",
                conversation_id="cnv_telemetry",
                mode="coding_debug",
                message="How do I fix this retry loop?",
            )
        # The turn's transaction rolled back with it: no half-written turn.
        connection = await engine.runtime.open_connection()
        try:
            cursor = await connection.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                ("cnv_telemetry",),
            )
            assert (await cursor.fetchone())[0] == 0
        finally:
            await connection.close()
    finally:
        await engine.close()


# ---------------------------------------------------------------------------
# Retrieve-only (get_context) turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_context_retry_reuses_one_event_with_the_newest_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retried get_context must not double-count in the ledger.

    ``get_context`` is idempotent on ``message_id``: the second call returns the
    same message row rather than creating a second one. The retrieval event has
    to inherit that, or every aggregate over this table over-counts on exactly
    the path the API is designed to make retry-safe. Retrieval genuinely re-ran,
    so the surviving row carries the newest measurement.
    """
    engine = await _engine(monkeypatch)
    try:
        first = await engine.get_context(
            user_id="usr_telemetry",
            conversation_id="cnv_telemetry",
            message="Please help me debug this retry loop.",
            message_id="msg_retried",
        )
        second = await engine.get_context(
            user_id="usr_telemetry",
            conversation_id="cnv_telemetry",
            message="Please help me debug this retry loop.",
            message_id="msg_retried",
        )
        rows = await _turn_rows(engine)
        connection = await engine.runtime.open_connection()
        try:
            cursor = await connection.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE id = ?",
                ("msg_retried",),
            )
            message_count = (await cursor.fetchone())["total"]
        finally:
            await connection.close()
    finally:
        await engine.close()

    # One message, therefore one event: the two writes agree with each other.
    assert message_count == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["turn_surface"] == TurnSurface.CONTEXT.value
    assert first.retrieval_event_id == second.retrieval_event_id
    assert row["id"] == second.retrieval_event_id
    # The surviving row describes the SECOND run, not the first.
    assert row["retrieval_duration_ms"] == pytest.approx(
        second.retrieval_duration_ms
    )
    assert row["retrieval_duration_ms"] != pytest.approx(first.retrieval_duration_ms)


@pytest.mark.asyncio
async def test_context_surface_counts_its_retrieval_stage_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retrieve-only surface is the only one metered by SidecarService.

    With a corpus to search, retrieval spends real provider round-trips (need
    detection, applicability scoring). Those are Atagia's cost on a call whose
    reply the host runs itself, so they must land in the persisted counters --
    a non-negative assertion could never have shown that.
    """
    engine = await _engine(monkeypatch)
    try:
        connection = await engine.runtime.open_connection()
        try:
            await MemoryObjectRepository(
                connection, engine.runtime.clock
            ).create_memory_object(
                user_id="usr_telemetry",
                conversation_id="cnv_telemetry",
                assistant_mode_id="coding_debug",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.USER,
                canonical_text="The billing worker retry loop never backs off.",
                source_kind=MemorySourceKind.VERBATIM,
                confidence=0.9,
                privacy_level=0,
                memory_id="mem_retry_loop",
                payload={"writer_kind": "manual"},
            )
        finally:
            await connection.close()

        result = await engine.get_context(
            user_id="usr_telemetry",
            conversation_id="cnv_telemetry",
            message="Please help me debug this retry loop.",
        )
        rows = await _turn_rows(engine)
    finally:
        await engine.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == result.retrieval_event_id
    assert row["turn_surface"] == TurnSurface.CONTEXT.value
    assert row["llm_total_calls"] >= 1
    assert row["llm_failed_calls"] == 0
    assert row["llm_total_latency_ms"] > 0.0
    by_purpose = row["llm_by_purpose_json"]
    # Need detection is a retrieval-stage call, so it is inside this count.
    assert any(
        purpose.startswith("need_detection") for purpose in by_purpose
    ), by_purpose
    # The host owns the reply on this surface, so it is never in the count.
    assert "chat_reply" not in by_purpose
    assert sum(usage["calls"] for usage in by_purpose.values()) == row["llm_total_calls"]
    assert sum(
        usage["latency_ms"] for usage in by_purpose.values()
    ) == pytest.approx(row["llm_total_latency_ms"])
    # CS-1.4: the count has to be in the TRACE too, not only the typed columns.
    # This surface used to persist "llm_call_metrics": null on exactly the rows
    # whose retrieval demonstrably made calls.
    metrics = row["outcome_json"]["llm_call_metrics"]
    assert metrics is not None
    assert metrics["total_calls"] == row["llm_total_calls"]
    assert metrics["by_purpose"] == by_purpose
    trace_metrics = row["outcome_json"]["retrieval_trace"]["llm_call_metrics"]
    assert trace_metrics == metrics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "surface_call",
    ["chat", "context"],
)
async def test_turn_duration_includes_the_wait_for_the_per_user_cache_guard(
    monkeypatch: pytest.MonkeyPatch,
    surface_call: str,
) -> None:
    """The turn clock starts BEFORE the guard, so contention is measured.

    ``user_cache_guard`` is a per-user lock held for a whole turn, with a 30s
    acquire timeout. Starting the clock inside it put the queueing wait outside
    the measurement, so a caller who waited most of a second saw a turn reported
    as tens of milliseconds -- the exact latency this telemetry exists to show.
    """
    stall_seconds = 0.15
    original_guard = ContextCacheService.user_cache_guard

    @asynccontextmanager
    async def stalling_guard(
        self: ContextCacheService, user_id: str
    ) -> AsyncIterator[bool]:
        # Stand in for waiting behind another same-user turn already holding it.
        await asyncio.sleep(stall_seconds)
        async with original_guard(self, user_id) as acquired:
            yield acquired

    monkeypatch.setattr(ContextCacheService, "user_cache_guard", stalling_guard)
    engine = await _engine(monkeypatch)
    try:
        if surface_call == "chat":
            await engine.chat(
                user_id="usr_telemetry",
                conversation_id="cnv_telemetry",
                message="Please help me debug this retry loop.",
            )
        else:
            await engine.get_context(
                user_id="usr_telemetry",
                conversation_id="cnv_telemetry",
                message="Please help me debug this retry loop.",
            )
        rows = await _turn_rows(engine)
    finally:
        await engine.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["turn_to_event_write_wall_ms"] >= stall_seconds * 1000.0
    # The two clocks stay separate: lock wait is caller-visible turn time, not
    # retrieval work. Folding it into retrieval_duration_ms too would report a
    # turn that spent its time queueing as having spent it retrieving, and would
    # make turn_to_event_write_wall_ms - retrieval_duration_ms meaningless.
    assert row["retrieval_duration_ms"] < stall_seconds * 1000.0
    assert row["retrieval_duration_ms"] <= row["turn_to_event_write_wall_ms"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_TELEMETRY_COLUMNS = (
    "turn_surface",
    "turn_to_event_write_wall_ms",
    "retrieval_duration_ms",
    "llm_total_calls",
    "llm_failed_calls",
    "llm_total_latency_ms",
    "llm_by_purpose_json",
    "stage_timings_ms_json",
)


@pytest.mark.asyncio
async def test_turn_telemetry_migrations_apply_on_a_fresh_database() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        cursor = await connection.execute("PRAGMA table_info(retrieval_events)")
        columns = {row["name"] for row in await cursor.fetchall()}
        assert set(_TELEMETRY_COLUMNS) <= columns
        # 0070 replaced the calls-only breakdown; leaving the old column behind
        # would be a trap for the next reader, so it must be gone.
        assert "llm_calls_by_purpose_json" not in columns
        cursor = await connection.execute("PRAGMA table_info(memory_objects)")
        memory_columns = {row["name"] for row in await cursor.fetchall()}
        # Both ends of the ingest interval. 0071 added the right-hand one after
        # 0069 shipped with created_at standing in for it; a schema carrying
        # only the arrival stamp cannot express ingest latency honestly.
        assert "source_message_created_at" in memory_columns
        assert "queryable_at" in memory_columns
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("idx_retrieval_events_turn_surface",),
        )
        assert await cursor.fetchone() is not None
        # The retry-safety invariant is enforced by the schema, not by the
        # writer remembering to look first.
        cursor = await connection.execute(
            "PRAGMA index_list(retrieval_events)"
        )
        indexes = {row["name"]: row for row in await cursor.fetchall()}
        context_index = indexes["uq_retrieval_events_context_request"]
        assert context_index["unique"] == 1
        assert context_index["partial"] == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_context_request_uniqueness_is_scoped_to_the_context_surface() -> None:
    """Only ``get_context`` is retry-deduplicated; the other surfaces are not.

    A proxy turn's row starts life on the ``context`` surface and is upgraded in
    place, and chat writes its own row per request message. A full-table unique
    index would reject legitimate rows, so the constraint carries the surface
    predicate and this test is what keeps it that way.
    """
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_minimal_namespace(connection)
        for surface in (
            TurnSurface.CHAT,
            TurnSurface.PROXY_COMPLETION,
            TurnSurface.PROXY_STREAM,
        ):
            for suffix in ("a", "b"):
                await _insert_raw_event(
                    connection,
                    event_id=f"ret_{surface.value}_{suffix}",
                    surface=surface,
                )
        await _insert_raw_event(
            connection, event_id="ret_context_a", surface=TurnSurface.CONTEXT
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            await _insert_raw_event(
                connection, event_id="ret_context_b", surface=TurnSurface.CONTEXT
            )
    finally:
        await connection.close()


_SEED_TIMESTAMP = "2026-07-24T10:00:00+00:00"


async def _seed_minimal_namespace(connection: aiosqlite.Connection) -> None:
    """Insert the user/mode/conversation/message a retrieval event references."""
    await connection.execute(
        "INSERT INTO users(id, created_at, updated_at) VALUES ('usr_ix', ?, ?)",
        (_SEED_TIMESTAMP, _SEED_TIMESTAMP),
    )
    await connection.execute(
        """
        INSERT INTO assistant_modes(
            id, display_name, prompt_hash, memory_policy_json, created_at, updated_at
        ) VALUES ('coding_debug', 'Coding Debug', 'hash', '{}', ?, ?)
        """,
        (_SEED_TIMESTAMP, _SEED_TIMESTAMP),
    )
    await connection.execute(
        """
        INSERT INTO conversations(
            id, user_id, assistant_mode_id, title, status, created_at, updated_at
        ) VALUES ('cnv_ix', 'usr_ix', 'coding_debug', 'Index', 'active', ?, ?)
        """,
        (_SEED_TIMESTAMP, _SEED_TIMESTAMP),
    )
    await connection.execute(
        """
        INSERT INTO messages(id, conversation_id, role, seq, text, created_at)
        VALUES ('msg_ix', 'cnv_ix', 'user', 1, 'Index probe', ?)
        """,
        (_SEED_TIMESTAMP,),
    )
    await connection.commit()


async def _insert_raw_event(
    connection: aiosqlite.Connection,
    *,
    event_id: str,
    surface: TurnSurface,
) -> None:
    """Insert one event straight through SQL, bypassing repository guards."""
    await connection.execute(
        """
        INSERT INTO retrieval_events(
            id, user_id, conversation_id, request_message_id, assistant_mode_id,
            retrieval_plan_json, selected_memory_ids_json, context_view_json,
            outcome_json, created_at, turn_surface
        ) VALUES (?, 'usr_ix', 'cnv_ix', 'msg_ix', 'coding_debug',
                  '{}', '[]', '{}', '{}', ?, ?)
        """,
        (event_id, _SEED_TIMESTAMP, surface.value),
    )
    await connection.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("bootstrap_version", _TELEMETRY_BOOTSTRAP_VERSIONS)
async def test_turn_telemetry_migrations_upgrade_an_existing_database(
    tmp_path: Path,
    bootstrap_version: int,
) -> None:
    """Both entry points into the telemetry columns must land on the same schema.

    A database can arrive here from BEFORE any telemetry column existed (68) or
    from the intermediate state that had the calls-only breakdown (69). Only the
    second exercises 0070's drop-and-replace, and it is the state every
    developer database on this branch is actually in.
    """
    bootstrap = tmp_path / f"migrations-through-{bootstrap_version:04d}"
    bootstrap.mkdir()
    manager = MigrationManager(MIGRATIONS_DIR)
    for migration in manager.discover():
        if migration.version <= bootstrap_version:
            copy2(migration.path, bootstrap / migration.path.name)
    database_path = tmp_path / "telemetry-upgrade.db"
    connection = await initialize_database(str(database_path), bootstrap)
    try:
        await connection.execute(
            "INSERT INTO users(id, created_at, updated_at) VALUES ('usr_1', ?, ?)",
            ("2026-07-20T10:00:00+00:00", "2026-07-20T10:00:00+00:00"),
        )
        await connection.execute(
            """
            INSERT INTO assistant_modes(
                id, display_name, prompt_hash, memory_policy_json, created_at, updated_at
            ) VALUES ('coding_debug', 'Coding Debug', 'hash', '{}', ?, ?)
            """,
            ("2026-07-20T10:00:00+00:00", "2026-07-20T10:00:00+00:00"),
        )
        await connection.execute(
            """
            INSERT INTO conversations(
                id, user_id, assistant_mode_id, title, status, created_at, updated_at
            ) VALUES ('cnv_1', 'usr_1', 'coding_debug', 'Legacy', 'active', ?, ?)
            """,
            ("2026-07-20T10:00:00+00:00", "2026-07-20T10:00:00+00:00"),
        )
        await connection.execute(
            """
            INSERT INTO messages(id, conversation_id, role, seq, text, created_at)
            VALUES ('msg_1', 'cnv_1', 'user', 1, 'Legacy prompt', ?)
            """,
            ("2026-07-20T10:00:00+00:00",),
        )
        await connection.execute(
            """
            INSERT INTO retrieval_events(
                id, user_id, conversation_id, request_message_id, assistant_mode_id,
                retrieval_plan_json, selected_memory_ids_json, context_view_json,
                outcome_json, created_at
            ) VALUES ('ret_legacy', 'usr_1', 'cnv_1', 'msg_1', 'coding_debug',
                      '{}', '[]', '{}', '{}', ?)
            """,
            ("2026-07-20T10:00:00+00:00",),
        )
        if bootstrap_version >= 69:
            # A row written under the intermediate schema: it recorded REAL
            # calls per purpose but never measured latency per purpose. Two
            # purposes, so the backfill has to carry a map rather than a single
            # entry it could have special-cased.
            await connection.execute(
                """
                UPDATE retrieval_events
                SET llm_total_calls = 5,
                    llm_calls_by_purpose_json =
                        '{"chat_reply": 3, "need_detection": 2}'
                WHERE id = 'ret_legacy'
                """
            )
        await connection.commit()
    finally:
        await connection.close()

    connection = await initialize_database(str(database_path), MIGRATIONS_DIR)
    try:
        cursor = await connection.execute(
            "SELECT * FROM retrieval_events WHERE id = 'ret_legacy'"
        )
        legacy = await cursor.fetchone()
        # Pre-migration rows can only have come from a chat turn, so the surface
        # backfill is exact, while every measurement stays NULL rather than
        # inventing a zero that would pollute averages.
        assert legacy["turn_surface"] == TurnSurface.CHAT.value
        assert legacy["turn_to_event_write_wall_ms"] is None
        assert legacy["retrieval_duration_ms"] is None
        assert legacy["stage_timings_ms_json"] is None
        # The replaced column is gone, not shadowed by the new one.
        assert "llm_calls_by_purpose_json" not in legacy.keys()
        if bootstrap_version >= 69:
            # The counts SURVIVE the column swap: they are real measurements and
            # 0070's inability to carry latency honestly was never a reason to
            # destroy them. The latency is an explicit null -- 0069's "written
            # before this migration" convention one level down -- and not the
            # fabricated 0.0 that 0069 itself refused.
            assert json.loads(legacy["llm_by_purpose_json"]) == {
                "chat_reply": {"calls": 3, "latency_ms": None},
                "need_detection": {"calls": 2, "latency_ms": None},
            }
            assert legacy["llm_total_calls"] == 5
        else:
            # A pre-0069 row never had a breakdown at all, so NULL stays NULL:
            # "nothing was ever recorded" must not become "an empty breakdown
            # was recorded".
            assert legacy["llm_by_purpose_json"] is None
            assert legacy["llm_total_calls"] is None

        events = RetrievalEventRepository(connection, _FrozenIsoClock())
        created = await events.create_event(
            {
                "user_id": "usr_1",
                "conversation_id": "cnv_1",
                "request_message_id": "msg_1",
                "assistant_mode_id": "coding_debug",
                "retrieval_plan_json": {},
                "selected_memory_ids_json": [],
                "context_view_json": {},
                "outcome_json": {},
            },
            telemetry=sample_turn_telemetry(),
        )
        assert created["turn_surface"] == TurnSurface.CHAT.value
        assert created["llm_total_calls"] == 3
    finally:
        await connection.close()


class _FrozenIsoClock:
    def now(self) -> datetime:
        return datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Telemetry payload validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"turn_to_event_write_wall_ms": -1.0}, "durations cannot be negative"),
        ({"retrieval_duration_ms": -0.5}, "durations cannot be negative"),
        ({"llm_total_calls": -1}, "call counts cannot be negative"),
        ({"llm_failed_calls": -1}, "call counts cannot be negative"),
        ({"llm_total_latency_ms": -2.0}, "LLM latency cannot be negative"),
        (
            {"llm_total_calls": 1, "llm_failed_calls": 2},
            "failed calls exceed total calls",
        ),
        # A retrieval slice cannot outlast the turn that contains it.
        (
            {"retrieval_duration_ms": 99999.0},
            "retrieval duration exceeds the turn",
        ),
        # Every counted call must be attributable to a purpose, or the breakdown
        # is not a breakdown.
        (
            {
                "llm_by_purpose": {
                    "chat_reply": LLMPurposeMetricsTrace(calls=77, latency_ms=8.0)
                }
            },
            "does not account for every call",
        ),
        (
            {"llm_by_purpose": {}},
            "does not account for every call",
        ),
        # CS-1.5: the per-purpose latency is a breakdown of the turn's provider
        # latency, so it can never claim more wall time than the turn measured.
        (
            {
                "llm_by_purpose": {
                    "chat_reply": LLMPurposeMetricsTrace(calls=2, latency_ms=9.0)
                }
            },
            "by-purpose latency exceeds the measured total",
        ),
        # A null latency is representable so migration 0070's carried-forward
        # counts can be read back, but it is only ever a pre-migration artifact:
        # a writer that supplies one has failed to measure and must fail fast.
        (
            {
                "llm_by_purpose": {
                    "chat_reply": LLMPurposeMetricsTrace(calls=2, latency_ms=None)
                }
            },
            "requires a measured latency for every purpose",
        ),
        (
            {"stage_timings_ms": {"planning": -1.0}},
            "stage timings cannot be negative",
        ),
    ],
)
def test_turn_telemetry_rejects_impossible_measurements(
    overrides: dict[str, Any],
    message: str,
) -> None:
    fields: dict[str, Any] = {
        "surface": TurnSurface.CHAT,
        "turn_to_event_write_wall_ms": 10.0,
        "retrieval_duration_ms": 4.0,
        "llm_total_calls": 2,
        "llm_failed_calls": 0,
        "llm_total_latency_ms": 8.0,
        "llm_by_purpose": {
            "chat_reply": LLMPurposeMetricsTrace(calls=2, latency_ms=8.0)
        },
        "stage_timings_ms": {"planning": 1.0},
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        TurnTelemetry(**fields)
