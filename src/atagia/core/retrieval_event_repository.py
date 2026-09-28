"""Repositories for retrieval traces, feedback, and admin audit logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.ids import generate_prefixed_id, new_retrieval_id
from atagia.core.repositories import (
    BaseRepository,
    MemoryObjectRepository,
    _encode_json,
)
from atagia.models.schemas_memory import (
    LLMCallMetricsTrace,
    LLMPurposeMetricsTrace,
    TurnSurface,
)

# Per-purpose latency is summed from the same float additions as the total, so
# the two can differ in the last bits of a float without anything being wrong.
_LATENCY_SUM_TOLERANCE_MS: float = 1e-6


@dataclass(frozen=True, slots=True)
class TurnTelemetry:
    """Per-turn latency and LLM-call measurements stored as queryable columns.

    Every field is required so a turn cannot be persisted with a silently
    missing measurement; a NULL column therefore means "written before the
    telemetry migration" and never "the writer forgot".

    ``turn_to_event_write_wall_ms`` is named for its endpoint because that is
    where it stops: it runs from the start of the turn to the instant this row
    is written, which is BEFORE the rest of the terminal transaction, the
    commit, cache publication, job dispatch and result assembly. It is not the
    turn's total duration and the difference against ``retrieval_duration_ms``
    is not "everything after retrieval". Migration 0072 records the measured
    size of the excluded tail per surface and why the span was not extended to
    cover it.

    ``llm_by_purpose`` holds calls AND provider latency per purpose in a single
    map (CS-1.5), so "which stage spent the wall time" is answerable from the
    same record as "which stage spent the calls" and the two can never disagree
    about the set of purposes.

    ``__post_init__`` enforces the relationships the columns cannot express on
    their own: retrieval is a slice of its turn, the by-purpose breakdown
    accounts for every counted call, every purpose carries a measured latency,
    the breakdown never claims more latency than the turn measured, and no
    timing runs backwards. Every production call site already satisfies them, so
    checking here turns a convention into a guarantee a reader can rely on.

    A null ``latency_ms`` inside ``llm_by_purpose`` is representable on the READ
    path -- migration 0070 carried the 0069 call counts forward with the latency
    explicitly absent -- but never writable: a writer that supplies one fails
    here rather than persisting an unmeasured turn.
    """

    surface: TurnSurface
    turn_to_event_write_wall_ms: float
    retrieval_duration_ms: float
    llm_total_calls: int
    llm_failed_calls: int
    llm_total_latency_ms: float
    llm_by_purpose: dict[str, LLMPurposeMetricsTrace]
    stage_timings_ms: dict[str, float]

    def __post_init__(self) -> None:
        if self.turn_to_event_write_wall_ms < 0.0 or self.retrieval_duration_ms < 0.0:
            raise ValueError("Turn telemetry durations cannot be negative")
        if self.retrieval_duration_ms > self.turn_to_event_write_wall_ms:
            raise ValueError(
                "Turn telemetry retrieval duration exceeds the turn that contains "
                f"it: retrieval_duration_ms={self.retrieval_duration_ms} > "
                f"turn_to_event_write_wall_ms={self.turn_to_event_write_wall_ms}"
            )
        if self.llm_total_calls < 0 or self.llm_failed_calls < 0:
            raise ValueError("Turn telemetry LLM call counts cannot be negative")
        if self.llm_failed_calls > self.llm_total_calls:
            raise ValueError("Turn telemetry failed calls exceed total calls")
        by_purpose_calls = sum(usage.calls for usage in self.llm_by_purpose.values())
        if by_purpose_calls != self.llm_total_calls:
            raise ValueError(
                "Turn telemetry by-purpose breakdown does not account for every "
                f"call: llm_total_calls={self.llm_total_calls} but "
                f"llm_by_purpose sums to {by_purpose_calls} calls"
            )
        if self.llm_total_latency_ms < 0.0:
            raise ValueError("Turn telemetry LLM latency cannot be negative")
        by_purpose_latency_ms = 0.0
        for purpose, usage in self.llm_by_purpose.items():
            if usage.latency_ms is None:
                raise ValueError(
                    "Turn telemetry requires a measured latency for every "
                    f"purpose, but '{purpose}' carries none; a null latency_ms "
                    "exists only on rows migration 0070 carried forward from "
                    "the calls-only breakdown"
                )
            by_purpose_latency_ms += usage.latency_ms
        if by_purpose_latency_ms > self.llm_total_latency_ms + _LATENCY_SUM_TOLERANCE_MS:
            raise ValueError(
                "Turn telemetry by-purpose latency exceeds the measured total: "
                f"llm_total_latency_ms={self.llm_total_latency_ms} but "
                f"llm_by_purpose sums to {by_purpose_latency_ms}"
            )
        negative_stages = {
            stage: value
            for stage, value in self.stage_timings_ms.items()
            if value < 0.0
        }
        if negative_stages:
            raise ValueError(
                f"Turn telemetry stage timings cannot be negative: {negative_stages}"
            )

    def llm_call_metrics(self) -> LLMCallMetricsTrace:
        """Return the trace-shaped view of this turn's LLM call metrics.

        The typed columns and the trace payload are derived from ONE object, so
        a surface cannot persist a count in one place and a different count in
        the other.
        """
        return LLMCallMetricsTrace(
            total_calls=self.llm_total_calls,
            failed_calls=self.llm_failed_calls,
            total_latency_ms=self.llm_total_latency_ms,
            by_purpose=dict(self.llm_by_purpose),
        )

    def column_values(self) -> tuple[Any, ...]:
        """Return the telemetry column values in schema order."""
        return (
            self.surface.value,
            float(self.turn_to_event_write_wall_ms),
            float(self.retrieval_duration_ms),
            int(self.llm_total_calls),
            int(self.llm_failed_calls),
            float(self.llm_total_latency_ms),
            _encode_json(
                {
                    purpose: usage.model_dump(mode="json")
                    for purpose, usage in self.llm_by_purpose.items()
                }
            ),
            _encode_json(dict(self.stage_timings_ms)),
        )


def _mutable_event_values(event: dict[str, Any]) -> tuple[Any, ...]:
    """Return the rewritable retrieval-event column values in schema order.

    Shared by the insert and the ``context``-surface overwrite so the two
    statements cannot drift apart when a column is added. The identity columns
    (``id``, ``user_id``, ``conversation_id``, ``request_message_id``) are
    deliberately absent: they are what an overwrite matches on, never what it
    changes.
    """
    return (
        event.get("response_message_id"),
        event["assistant_mode_id"],
        event.get("user_persona_id"),
        event.get("platform_id"),
        event.get("character_id"),
        event.get("mode"),
        1 if event.get("incognito") else 0,
        1 if event.get("remember_across_chats", True) else 0,
        1 if event.get("remember_across_devices", True) else 0,
        _encode_json(event["retrieval_plan_json"]),
        _encode_json(event.get("selected_memory_ids_json", [])),
        _encode_json(event.get("context_view_json", {})),
        _encode_json(event.get("outcome_json", {})),
    )


class MemoryFeedbackOwnershipError(ValueError):
    """Raised when feedback references an event or memory outside the user's scope."""


class MemoryFeedbackMismatchError(ValueError):
    """Raised when feedback references a memory not returned in the retrieval event."""


class RetrievalEventRepository(BaseRepository):
    """Persistence operations for retrieval event traces."""

    async def create_event(
        self,
        event: dict[str, Any],
        *,
        telemetry: TurnTelemetry,
        commit: bool = True,
    ) -> dict[str, Any]:
        event_id = str(event.get("id") or new_retrieval_id())
        timestamp = str(event.get("created_at") or self._timestamp())
        await self._connection.execute(
            """
            INSERT INTO retrieval_events(
                id,
                user_id,
                conversation_id,
                request_message_id,
                response_message_id,
                assistant_mode_id,
                user_persona_id,
                platform_id,
                character_id,
                mode,
                incognito,
                remember_across_chats,
                remember_across_devices,
                retrieval_plan_json,
                selected_memory_ids_json,
                context_view_json,
                outcome_json,
                created_at,
                turn_surface,
                turn_to_event_write_wall_ms,
                retrieval_duration_ms,
                llm_total_calls,
                llm_failed_calls,
                llm_total_latency_ms,
                llm_by_purpose_json,
                stage_timings_ms_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event["user_id"],
                event["conversation_id"],
                event["request_message_id"],
                *_mutable_event_values(event),
                timestamp,
                *telemetry.column_values(),
            ),
        )
        if commit:
            await self._connection.commit()
        created = await self.get_event(event_id, str(event["user_id"]))
        if created is None:
            raise RuntimeError("Failed to create retrieval event row")
        return created

    async def upsert_context_event(
        self,
        event: dict[str, Any],
        *,
        telemetry: TurnTelemetry,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Write the retrieval event for one ``get_context`` call, retry-safe.

        ``get_context`` is idempotent on ``message_id``: re-posting the same
        message returns the same message row instead of creating a second one.
        Its retrieval event has to inherit that property, or every aggregate
        over this ledger over-counts on exactly the path the API is designed to
        make retry-safe. The retry did genuinely re-run retrieval, so the newest
        measurement replaces the previous one on the SAME row instead of minting
        a second ``ret_*`` id.

        The lookup runs under the caller's ``BEGIN IMMEDIATE`` fence, and the
        partial unique index on ``(user_id, conversation_id, request_message_id)
        WHERE turn_surface = 'context'`` is the backstop: a concurrent writer
        that slipped between the read and the insert fails the transaction
        rather than duplicating the row.
        """
        if telemetry.surface is not TurnSurface.CONTEXT:
            raise ValueError(
                "upsert_context_event is scoped to the context surface, got "
                f"{telemetry.surface.value}"
            )
        user_id = str(event["user_id"])
        existing = await self._fetch_one(
            """
            SELECT id
            FROM retrieval_events
            WHERE user_id = ?
              AND conversation_id = ?
              AND request_message_id = ?
              AND turn_surface = ?
            """,
            (
                user_id,
                event["conversation_id"],
                event["request_message_id"],
                TurnSurface.CONTEXT.value,
            ),
        )
        if existing is None:
            return await self.create_event(event, telemetry=telemetry, commit=commit)

        event_id = str(existing["id"])
        timestamp = str(event.get("created_at") or self._timestamp())
        await self._connection.execute(
            """
            UPDATE retrieval_events
            SET response_message_id = ?,
                assistant_mode_id = ?,
                user_persona_id = ?,
                platform_id = ?,
                character_id = ?,
                mode = ?,
                incognito = ?,
                remember_across_chats = ?,
                remember_across_devices = ?,
                retrieval_plan_json = ?,
                selected_memory_ids_json = ?,
                context_view_json = ?,
                outcome_json = ?,
                created_at = ?,
                turn_surface = ?,
                turn_to_event_write_wall_ms = ?,
                retrieval_duration_ms = ?,
                llm_total_calls = ?,
                llm_failed_calls = ?,
                llm_total_latency_ms = ?,
                llm_by_purpose_json = ?,
                stage_timings_ms_json = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                *_mutable_event_values(event),
                timestamp,
                *telemetry.column_values(),
                event_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        updated = await self.get_event(event_id, user_id)
        if updated is None:
            raise RuntimeError("Failed to overwrite context retrieval event row")
        return updated

    async def complete_turn_telemetry(
        self,
        event_id: str,
        user_id: str,
        *,
        response_message_id: str,
        telemetry: TurnTelemetry,
        commit: bool = True,
    ) -> None:
        """Upgrade a retrieval-scoped event into a completed turn's telemetry.

        The proxy writes its retrieval event before the reply exists, so the
        response message and the full-turn measurements are only known at the
        terminal commit. Missing the row is a bug, not a tolerable state: the
        caller passes an id it has just read back.

        ``outcome_json`` is rewritten alongside the typed columns. The row was
        written by the sidecar retrieval, whose trace carries the retrieval-only
        call metrics; leaving it untouched made the proxy surfaces persist a
        trace that read as "no calls" on exactly the rows where the reply
        round-trip did happen. Both the outcome-level metrics and the copy
        nested inside the retrieval trace are refreshed from the turn telemetry,
        so every consumer of the row sees the same numbers as the columns.
        """
        existing = await self.get_event(event_id, user_id)
        if existing is None:
            raise RuntimeError(
                f"Retrieval event {event_id} is not available for turn telemetry"
            )
        raw_outcome = existing.get("outcome_json")
        outcome = dict(raw_outcome) if isinstance(raw_outcome, dict) else {}
        call_metrics = telemetry.llm_call_metrics().model_dump(mode="json")
        outcome["llm_call_metrics"] = call_metrics
        trace = outcome.get("retrieval_trace")
        if isinstance(trace, dict):
            outcome["retrieval_trace"] = {**trace, "llm_call_metrics": call_metrics}
        cursor = await self._connection.execute(
            """
            UPDATE retrieval_events
            SET response_message_id = ?,
                outcome_json = ?,
                turn_surface = ?,
                turn_to_event_write_wall_ms = ?,
                retrieval_duration_ms = ?,
                llm_total_calls = ?,
                llm_failed_calls = ?,
                llm_total_latency_ms = ?,
                llm_by_purpose_json = ?,
                stage_timings_ms_json = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                response_message_id,
                _encode_json(outcome),
                *telemetry.column_values(),
                event_id,
                user_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(
                f"Retrieval event {event_id} is not available for turn telemetry"
            )
        if commit:
            await self._connection.commit()

    async def get_event(self, event_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM retrieval_events
            WHERE id = ?
              AND user_id = ?
            """,
            (event_id, user_id),
        )

    async def list_events(
        self,
        user_id: str,
        conversation_id: str | None,
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if conversation_id is None:
            return await self._fetch_all(
                """
                SELECT *
                FROM retrieval_events
                WHERE user_id = ?
                ORDER BY created_at DESC, id ASC
                LIMIT ?
                OFFSET ?
                """,
                (user_id, limit, offset),
            )
        return await self._fetch_all(
            """
            SELECT *
            FROM retrieval_events
            WHERE user_id = ?
              AND conversation_id = ?
            ORDER BY created_at DESC, id ASC
            LIMIT ?
            OFFSET ?
            """,
            (user_id, conversation_id, limit, offset),
        )

    async def list_events_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM retrieval_events
            WHERE user_id = ?
              AND conversation_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id, conversation_id),
        )

    async def update_outcome_fields(
        self,
        event_id: str,
        user_id: str,
        updates: dict[str, Any],
        *,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        event = await self.get_event(event_id, user_id)
        if event is None:
            return None
        outcome = event.get("outcome_json")
        current_outcome = dict(outcome) if isinstance(outcome, dict) else {}
        current_outcome.update(updates)
        await self._connection.execute(
            """
            UPDATE retrieval_events
            SET outcome_json = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                _encode_json(current_outcome),
                event_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        return await self.get_event(event_id, user_id)


class MemoryFeedbackRepository(BaseRepository):
    """Persistence operations for memory usefulness feedback."""

    async def create_feedback(
        self,
        retrieval_event_id: str | None,
        memory_id: str | None,
        user_id: str,
        feedback_type: str,
        score: float | None,
        metadata: dict[str, Any] | None = None,
        *,
        conversation_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        active_mind_id: str | None = None,
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        mode: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        selected_memory_ids: set[str] | None = None
        namespace_required = conversation_id is not None and platform_id is not None
        if retrieval_event_id is not None:
            event_row = await self._fetch_one(
                """
                SELECT *
                FROM retrieval_events
                WHERE id = ?
                  AND user_id = ?
                """,
                (retrieval_event_id, user_id),
            )
            if event_row is None:
                raise MemoryFeedbackOwnershipError(
                    f"Retrieval event {retrieval_event_id} does not belong to user {user_id}"
                )
            if namespace_required and (
                event_row.get("conversation_id") != conversation_id
                or event_row.get("user_persona_id") != user_persona_id
                or event_row.get("platform_id") != platform_id
                or event_row.get("character_id") != character_id
                or bool(event_row.get("incognito")) != bool(incognito)
            ):
                raise MemoryFeedbackOwnershipError(
                    f"Retrieval event {retrieval_event_id} is outside the requested namespace"
                )
            raw_selected_ids = event_row.get("selected_memory_ids_json") or []
            if isinstance(raw_selected_ids, list):
                selected_memory_ids = {str(item) for item in raw_selected_ids}
            else:
                selected_memory_ids = set()
        if memory_id is not None:
            if namespace_required:
                memory = await MemoryObjectRepository(
                    self._connection,
                    self._clock,
                ).get_visible_memory_object(
                    memory_id,
                    user_id,
                    conversation_id=conversation_id,
                    user_persona_id=user_persona_id,
                    platform_id=platform_id,
                    character_id=character_id,
                    incognito=incognito,
                    remember_across_chats=remember_across_chats,
                    remember_across_devices=remember_across_devices,
                    active_mind_id=active_mind_id,
                    mind_topology=mind_topology,
                    active_embodiment_id=active_embodiment_id,
                    active_realm_id=active_realm_id,
                    sensitivity_gates_enabled=True,
                )
                if memory is None:
                    raise MemoryFeedbackOwnershipError(
                        f"Memory object {memory_id} is outside the requested namespace"
                    )
            else:
                cursor = await self._connection.execute(
                    """
                    SELECT 1
                    FROM memory_objects
                    WHERE id = ?
                      AND user_id = ?
                    """,
                    (memory_id, user_id),
                )
                if await cursor.fetchone() is None:
                    raise MemoryFeedbackOwnershipError(
                        f"Memory object {memory_id} does not belong to user {user_id}"
                    )
            if selected_memory_ids is not None and memory_id not in selected_memory_ids:
                raise MemoryFeedbackMismatchError(
                    f"Memory object {memory_id} was not selected in retrieval event {retrieval_event_id}"
                )

        feedback_id = generate_prefixed_id("fbk")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO memory_feedback_events(
                id,
                user_id,
                retrieval_event_id,
                memory_id,
                feedback_type,
                score,
                metadata_json,
                created_at,
                user_persona_id,
                platform_id,
                character_id,
                conversation_id,
                mode,
                incognito_snapshot,
                remember_across_chats_snapshot,
                remember_across_devices_snapshot,
                policy_snapshot_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                user_id,
                retrieval_event_id,
                memory_id,
                feedback_type,
                score,
                _encode_json(metadata),
                timestamp,
                user_persona_id,
                platform_id,
                character_id,
                conversation_id,
                mode,
                1 if incognito else 0,
                1 if remember_across_chats else 0,
                1 if remember_across_devices else 0,
                _encode_json(
                    {
                        "source": "memory_feedback",
                        "conversation_id": conversation_id,
                        "mode": mode,
                    }
                ),
            ),
        )
        if commit:
            await self._connection.commit()
        created = await self._fetch_one(
            """
            SELECT *
            FROM memory_feedback_events
            WHERE id = ?
              AND user_id = ?
            """,
            (feedback_id, user_id),
        )
        if created is None:
            raise RuntimeError("Failed to create memory feedback row")
        return created

    async def list_feedback(self, memory_id: str, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_feedback_events
            WHERE memory_id = ?
              AND user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (memory_id, user_id),
        )


class AdminAuditRepository(BaseRepository):
    """Persistence operations for lightweight admin read auditing."""

    async def create_audit_entry(
        self,
        *,
        admin_user_id: str,
        action: str,
        target_type: str,
        target_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        audit_id = generate_prefixed_id("aud")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO admin_audit_log(
                id,
                admin_user_id,
                action,
                target_type,
                target_id,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                admin_user_id,
                action,
                target_type,
                target_id,
                _encode_json(metadata),
                timestamp,
            ),
        )
        await self._connection.commit()
        created = await self._fetch_one(
            """
            SELECT *
            FROM admin_audit_log
            WHERE id = ?
            """,
            (audit_id,),
        )
        if created is None:
            raise RuntimeError("Failed to create admin audit row")
        return created

    async def list_entries(
        self,
        admin_user_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if admin_user_id is None:
            return await self._fetch_all(
                """
                SELECT *
                FROM admin_audit_log
                ORDER BY created_at ASC, _rowid ASC
                LIMIT ?
                """,
                (limit,),
            )
        return await self._fetch_all(
            """
            SELECT *
            FROM admin_audit_log
            WHERE admin_user_id = ?
            ORDER BY created_at ASC, _rowid ASC
            LIMIT ?
            """,
            (admin_user_id, limit),
        )


def build_logging_repositories(
    connection: aiosqlite.Connection,
    clock: Clock,
) -> tuple[RetrievalEventRepository, MemoryFeedbackRepository, AdminAuditRepository]:
    """Return the Step 10 logging repositories sharing one connection/clock."""
    return (
        RetrievalEventRepository(connection, clock),
        MemoryFeedbackRepository(connection, clock),
        AdminAuditRepository(connection, clock),
    )
