"""SQLite ownership, fencing, and replay records for proxy turn pairs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import hashlib
import secrets
from time import perf_counter
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence

from atagia.core.conversation_namespace import (
    ConversationNamespaceSnapshot,
    capture_conversation_namespace_snapshot,
    parse_conversation_namespace_snapshot,
    serialize_conversation_namespace_snapshot,
)
from atagia.core.job_run_repository import JobRunRepository
from atagia.core.presence_repository import PresenceRepository, presence_snapshot
from atagia.core.repositories import BaseRepository, MessageRepository
from atagia.core.retrieval_event_repository import (
    RetrievalEventRepository,
    TurnTelemetry,
)
from atagia.core.transcript_rebuild_repository import (
    TranscriptRebuildRepository,
    UserAvailabilitySnapshot,
)
from atagia.models.schemas_jobs import JobEnvelope
from atagia.services.errors import (
    ConversationNotFoundError,
    MessageIdConflictError,
    ProxyTurnConflictError,
    ProxyTurnInProgressError,
    ProxyTurnStaleOwnerError,
    SourceSequenceConflictError,
    TranscriptRebuildInProgressError,
)
from atagia.services.proxy_transcript import (
    PROXY_TRANSCRIPT_METADATA_KEY,
    canonical_json,
    idempotency_tool_projection,
    normalize_assistant_tool_calls,
)


DEFAULT_PROXY_TURN_LEASE_SECONDS = 30.0
DEFAULT_PROXY_RETRY_AFTER_SECONDS = 1.0
_ADMIT_CURRENT_NAMESPACE = object()


@dataclass(frozen=True, slots=True)
class ProxyTurnClaim:
    pair_id: str
    request_message_id: str
    response_message_id: str
    user_id: str
    conversation_id: str
    request_message_role: Literal["user", "tool"]
    response_source_seq: int | None
    claim_token: str
    owner_token: str
    owner_fence: int
    client_request_fingerprint: str
    lifecycle_epoch: str
    derivation_revision: int
    namespace_snapshot: ConversationNamespaceSnapshot
    final_provider_fingerprint: str | None = None
    emission_started: bool = False


@dataclass(frozen=True, slots=True)
class ProxyTurnReplay:
    pair_id: str
    request_message_id: str
    response_message_id: str
    response_row: dict[str, Any]
    replay_envelope: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProxyTurnReservation:
    claim: ProxyTurnClaim | None = None
    replay: ProxyTurnReplay | None = None
    created: bool = False
    takeover: bool = False


@dataclass(frozen=True, slots=True)
class ProxyTurnRequestMessage:
    message_id: str
    response_message_id: str
    user_id: str
    conversation_id: str
    role: Literal["user", "tool"]
    text: str
    metadata: dict[str, Any]
    source_seq: int | None = None
    response_source_seq: int | None = None
    occurred_at: str | None = None


@dataclass(frozen=True, slots=True)
class ProxyTurnResponseMessage:
    text: str
    metadata: dict[str, Any]
    source_seq: int | None = None
    occurred_at: str | None = None


@dataclass(frozen=True, slots=True)
class ProxyTurnTelemetry:
    """Per-turn telemetry applied inside the fenced terminal commit.

    ``retrieval_event_id`` is the row the turn's retrieval already wrote. When
    memory context failed open there is no such row and the turn still has to be
    counted, so ``fallback_event`` is inserted instead, carrying the same
    measurements with an empty retrieval plan and context view.

    ``turn_started_at`` is a ``perf_counter`` origin, not a wall-clock instant:
    the turn's wall time is re-measured at the moment the row is written so
    ``turn_to_event_write_wall_ms`` means the same span here as on the chat and
    context surfaces, instead of stopping short of the persistence work that
    this telemetry is part of.
    """

    telemetry: TurnTelemetry
    turn_started_at: float
    retrieval_event_id: str | None
    fallback_event: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProxyDurableJobInsert:
    stream_name: str
    target_backend: str
    envelope: JobEnvelope
    source_token_estimate: int | None
    size_bucket: str | None
    metadata: dict[str, Any]
    user_persona_id: str | None
    platform_id: str | None
    character_id: str | None
    incognito_snapshot: bool
    remember_across_chats_snapshot: bool
    remember_across_devices_snapshot: bool
    temporary_snapshot: bool
    purge_on_close_snapshot: bool
    policy_snapshot: dict[str, Any]


def proxy_turn_pair_id(
    *,
    user_id: str,
    conversation_id: str,
    request_message_id: str,
    response_message_id: str,
) -> str:
    material = "\0".join(
        (user_id, conversation_id, request_message_id, response_message_id)
    ).encode("utf-8")
    return f"ptr_{hashlib.sha256(material).hexdigest()[:40]}"


class ProxyTurnRepository(BaseRepository):
    """Own one generation and retain the canonical paired-message disposition."""

    async def reserve(
        self,
        *,
        request_message: ProxyTurnRequestMessage,
        client_request_fingerprint: str,
        expected_namespace_snapshot: ConversationNamespaceSnapshot
        | None
        | object = _ADMIT_CURRENT_NAMESPACE,
        lease_seconds: float = DEFAULT_PROXY_TURN_LEASE_SECONDS,
    ) -> ProxyTurnReservation:
        if request_message.message_id == request_message.response_message_id:
            raise ProxyTurnConflictError(
                "request_message_id and response_message_id must be distinct",
                code="proxy_message_id_pair_conflict",
            )
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        pair_id = proxy_turn_pair_id(
            user_id=request_message.user_id,
            conversation_id=request_message.conversation_id,
            request_message_id=request_message.message_id,
            response_message_id=request_message.response_message_id,
        )
        owner_token = f"pto_{secrets.token_hex(24)}"
        claim_token = f"ptc_{secrets.token_hex(24)}"
        committed = False
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            rebuild_repository = TranscriptRebuildRepository(
                self._connection,
                self._clock,
            )
            await rebuild_repository.require_user_available(request_message.user_id)
            run = await self.get_run(pair_id)
            if run is None:
                if expected_namespace_snapshot is None:
                    raise ProxyTurnConflictError(
                        "The proxy pair disappeared before namespace admission; retry the request",
                        code="proxy_pair_admission_changed",
                    )
                conversation = await self._active_conversation(
                    request_message.conversation_id,
                    request_message.user_id,
                )
                if conversation is None:
                    raise ConversationNotFoundError("Conversation not found for user")
                availability_snapshot = (
                    await rebuild_repository.capture_user_availability_snapshot(
                        request_message.user_id
                    )
                )
                await self._assert_ids_available_for_new_pair(request_message)
                await self._validate_tool_parent_link(request_message)
                source_presence_id = await self._source_presence_id_for_request(
                    conversation=conversation,
                    request_role=request_message.role,
                )
                namespace_snapshot = await self._current_namespace_snapshot(
                    user_id=request_message.user_id,
                    conversation_id=request_message.conversation_id,
                )
                if (
                    expected_namespace_snapshot is not _ADMIT_CURRENT_NAMESPACE
                    and namespace_snapshot != expected_namespace_snapshot
                ):
                    raise ProxyTurnConflictError(
                        "The conversation namespace changed before proxy admission; retry the request",
                        code="proxy_namespace_changed",
                    )
                now = self._clock.now()
                timestamp = now.isoformat()
                lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
                await self._connection.execute(
                    """
                    INSERT INTO proxy_turn_runs(
                        pair_id,
                        request_message_id,
                        response_message_id,
                        user_id,
                        conversation_id,
                        request_message_role,
                        response_source_seq,
                        state,
                        client_fingerprint_version,
                        client_request_fingerprint,
                        owner_token,
                        owner_fence,
                        lifecycle_epoch,
                        derivation_revision,
                        namespace_snapshot_json,
                        lease_expires_at,
                        created_at,
                        updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, 'generating', 1, ?, ?, 1, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        pair_id,
                        request_message.message_id,
                        request_message.response_message_id,
                        request_message.user_id,
                        request_message.conversation_id,
                        request_message.role,
                        request_message.response_source_seq,
                        client_request_fingerprint,
                        owner_token,
                        availability_snapshot.lifecycle_epoch,
                        availability_snapshot.derivation_revision,
                        canonical_json(
                            serialize_conversation_namespace_snapshot(
                                namespace_snapshot
                            )
                        ),
                        lease_expires_at,
                        timestamp,
                        timestamp,
                    ),
                )
                await self._insert_claims(
                    pair_id=pair_id,
                    request_message=request_message,
                    claim_token=claim_token,
                    timestamp=timestamp,
                )
                inserted_request = await self._insert_request_message(
                    conversation=conversation,
                    request_message=request_message,
                    claim_token=claim_token,
                    source_presence_id=source_presence_id,
                )
                request_source_seq = int(inserted_request["seq"])
                await self._validate_response_source_seq_for_new_pair(
                    request_message=request_message,
                    request_source_seq=request_source_seq,
                )
                await self._connection.execute(
                    """
                    UPDATE proxy_turn_runs
                    SET request_source_seq = ?
                    WHERE pair_id = ?
                    """,
                    (request_source_seq, pair_id),
                )
                await self._connection.commit()
                committed = True
                return ProxyTurnReservation(
                    claim=ProxyTurnClaim(
                        pair_id=pair_id,
                        request_message_id=request_message.message_id,
                        response_message_id=request_message.response_message_id,
                        user_id=request_message.user_id,
                        conversation_id=request_message.conversation_id,
                        request_message_role=request_message.role,
                        response_source_seq=request_message.response_source_seq,
                        claim_token=claim_token,
                        owner_token=owner_token,
                        owner_fence=1,
                        client_request_fingerprint=client_request_fingerprint,
                        lifecycle_epoch=availability_snapshot.lifecycle_epoch,
                        derivation_revision=(availability_snapshot.derivation_revision),
                        namespace_snapshot=namespace_snapshot,
                    ),
                    created=True,
                )

            claim_token = await self._validate_existing_pair(
                run=run,
                request_message=request_message,
                client_request_fingerprint=client_request_fingerprint,
            )
            state = str(run["state"])
            if state == "completed":
                replay = await self._completed_replay(run)
                await self._connection.commit()
                committed = True
                return ProxyTurnReservation(replay=replay)
            if state == "ambiguous_exposed":
                raise ProxyTurnConflictError(
                    "The prior stream may have been partially exposed; retry with new message IDs",
                    code="stream_retry_requires_new_ids",
                )
            if state == "final_fingerprint_conflict":
                error_code = str(run.get("error_code") or "")
                if error_code in {
                    "proxy_namespace_changed",
                    "proxy_namespace_snapshot_invalid",
                    "proxy_namespace_snapshot_missing",
                }:
                    raise ProxyTurnConflictError(
                        "The conversation namespace changed after this proxy turn was reserved; use new message IDs",
                        code="proxy_namespace_changed",
                    )
                raise ProxyTurnConflictError(
                    "The provider-bound request changed after an incomplete generation; use new message IDs",
                    code="proxy_final_request_changed",
                )

            availability_snapshot = self._source_snapshot(run)
            try:
                await rebuild_repository.require_user_availability_snapshot(
                    request_message.user_id,
                    availability_snapshot,
                )
            except TranscriptRebuildInProgressError as exc:
                raise ProxyTurnConflictError(
                    "Canonical memory sources changed after this proxy turn was reserved; use new message IDs",
                    code="proxy_source_revision_changed",
                ) from exc

            namespace_snapshot = await self._require_run_namespace_current(run=run)

            now = self._clock.now()
            timestamp = now.isoformat()
            lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
            lease_is_live = _timestamp_after(run.get("lease_expires_at"), now)
            if state == "emission_started":
                if lease_is_live:
                    raise ProxyTurnInProgressError(
                        "A generation for this message ID pair is already streaming",
                        code="request_in_progress",
                        retry_after_seconds=DEFAULT_PROXY_RETRY_AFTER_SECONDS,
                    )
                await self._connection.execute(
                    """
                    UPDATE proxy_turn_runs
                    SET state = 'ambiguous_exposed',
                        owner_token = NULL,
                        lease_expires_at = NULL,
                        ambiguous_at = ?,
                        error_code = 'stream_owner_lost',
                        error_message = 'Streaming ownership expired after emission began',
                        updated_at = ?
                    WHERE pair_id = ?
                      AND state = 'emission_started'
                      AND owner_fence = ?
                    """,
                    (timestamp, timestamp, pair_id, int(run["owner_fence"])),
                )
                await self._connection.commit()
                committed = True
                raise ProxyTurnConflictError(
                    "The prior stream may have been partially exposed; retry with new message IDs",
                    code="stream_retry_requires_new_ids",
                )
            if state != "generating":
                raise ProxyTurnConflictError(
                    f"Unsupported durable proxy turn state: {state}",
                    code="proxy_turn_state_conflict",
                )
            if lease_is_live:
                raise ProxyTurnInProgressError(
                    "A generation for this message ID pair is already in progress",
                    code="request_in_progress",
                    retry_after_seconds=DEFAULT_PROXY_RETRY_AFTER_SECONDS,
                )

            old_fence = int(run["owner_fence"])
            cursor = await self._connection.execute(
                """
                UPDATE proxy_turn_runs
                SET owner_token = ?,
                    owner_fence = owner_fence + 1,
                    lease_expires_at = ?,
                    error_code = NULL,
                    error_message = NULL,
                    updated_at = ?
                WHERE pair_id = ?
                  AND state = 'generating'
                  AND owner_fence = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                """,
                (
                    owner_token,
                    lease_expires_at,
                    timestamp,
                    pair_id,
                    old_fence,
                    timestamp,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise ProxyTurnInProgressError(
                    "Another request acquired this proxy turn",
                    code="request_in_progress",
                    retry_after_seconds=DEFAULT_PROXY_RETRY_AFTER_SECONDS,
                )
            await self._connection.commit()
            committed = True
            return ProxyTurnReservation(
                claim=ProxyTurnClaim(
                    pair_id=pair_id,
                    request_message_id=request_message.message_id,
                    response_message_id=request_message.response_message_id,
                    user_id=request_message.user_id,
                    conversation_id=request_message.conversation_id,
                    request_message_role=request_message.role,
                    response_source_seq=(
                        int(run["response_source_seq"])
                        if run.get("response_source_seq") is not None
                        else None
                    ),
                    claim_token=claim_token,
                    owner_token=owner_token,
                    owner_fence=old_fence + 1,
                    client_request_fingerprint=client_request_fingerprint,
                    lifecycle_epoch=availability_snapshot.lifecycle_epoch,
                    derivation_revision=availability_snapshot.derivation_revision,
                    namespace_snapshot=namespace_snapshot,
                    final_provider_fingerprint=run.get("final_provider_fingerprint"),
                ),
                takeover=True,
            )
        except Exception:
            if not committed and self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def establish_final_fingerprint(
        self,
        claim: ProxyTurnClaim,
        fingerprint: str,
        *,
        lease_seconds: float = DEFAULT_PROXY_TURN_LEASE_SECONDS,
    ) -> ProxyTurnClaim:
        """CAS the exact provider request before provider execution."""

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                self._connection,
                self._clock,
            ).require_user_availability_snapshot(
                claim.user_id,
                UserAvailabilitySnapshot(
                    lifecycle_epoch=claim.lifecycle_epoch,
                    derivation_revision=claim.derivation_revision,
                ),
            )
            run = await self.get_run(claim.pair_id)
            now = self._clock.now()
            self._require_current_owner(
                run,
                claim,
                allowed_states={"generating"},
                lease_boundary=now,
            )
            await self._require_claim_namespace_current(run=run, claim=claim)
            now = self._clock.now()
            self._require_current_owner(
                run,
                claim,
                allowed_states={"generating"},
                lease_boundary=now,
            )
            timestamp = now.isoformat()
            lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
            stored = run.get("final_provider_fingerprint") if run else None
            if stored is not None and str(stored) != fingerprint:
                cursor = await self._connection.execute(
                    """
                    UPDATE proxy_turn_runs
                    SET state = 'final_fingerprint_conflict',
                        owner_token = NULL,
                        lease_expires_at = NULL,
                        error_code = 'proxy_final_request_changed',
                        error_message = 'Provider-bound request fingerprint changed',
                        updated_at = ?
                    WHERE pair_id = ?
                      AND state = 'generating'
                      AND owner_token = ?
                      AND owner_fence = ?
                    """,
                    (
                        timestamp,
                        claim.pair_id,
                        claim.owner_token,
                        claim.owner_fence,
                    ),
                )
                if int(cursor.rowcount or 0) != 1:
                    raise self._stale_owner()
                await self._connection.commit()
                raise ProxyTurnConflictError(
                    "The provider-bound request changed after an incomplete generation; use new message IDs",
                    code="proxy_final_request_changed",
                )
            cursor = await self._connection.execute(
                """
                UPDATE proxy_turn_runs
                SET final_fingerprint_version = 1,
                    final_provider_fingerprint = COALESCE(final_provider_fingerprint, ?),
                    lease_expires_at = ?,
                    updated_at = ?
                WHERE pair_id = ?
                  AND state = 'generating'
                  AND owner_token = ?
                  AND owner_fence = ?
                  AND lease_expires_at > ?
                """,
                (
                    fingerprint,
                    lease_expires_at,
                    timestamp,
                    claim.pair_id,
                    claim.owner_token,
                    claim.owner_fence,
                    timestamp,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise self._stale_owner()
            await self._connection.commit()
            return replace(claim, final_provider_fingerprint=fingerprint)
        except Exception:
            if self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def mark_emission_started(
        self,
        claim: ProxyTurnClaim,
        *,
        lease_seconds: float = DEFAULT_PROXY_TURN_LEASE_SECONDS,
    ) -> ProxyTurnClaim:
        """Durably close the no-byte-yet race immediately before first SSE."""

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                self._connection,
                self._clock,
            ).require_user_availability_snapshot(
                claim.user_id,
                UserAvailabilitySnapshot(
                    lifecycle_epoch=claim.lifecycle_epoch,
                    derivation_revision=claim.derivation_revision,
                ),
            )
            run = await self.get_run(claim.pair_id)
            now = self._clock.now()
            self._require_current_owner(
                run,
                claim,
                allowed_states={"generating"},
                lease_boundary=now,
            )
            await self._require_claim_namespace_current(run=run, claim=claim)
            now = self._clock.now()
            self._require_current_owner(
                run,
                claim,
                allowed_states={"generating"},
                lease_boundary=now,
            )
            timestamp = now.isoformat()
            lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
            cursor = await self._connection.execute(
                """
                UPDATE proxy_turn_runs
                SET state = 'emission_started',
                    emission_started_at = ?,
                    lease_expires_at = ?,
                    updated_at = ?
                WHERE pair_id = ?
                  AND state = 'generating'
                  AND owner_token = ?
                  AND owner_fence = ?
                  AND lifecycle_epoch = ?
                  AND derivation_revision = ?
                  AND final_provider_fingerprint = ?
                  AND lease_expires_at > ?
                """,
                (
                    timestamp,
                    lease_expires_at,
                    timestamp,
                    claim.pair_id,
                    claim.owner_token,
                    claim.owner_fence,
                    claim.lifecycle_epoch,
                    claim.derivation_revision,
                    claim.final_provider_fingerprint,
                    timestamp,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise self._stale_owner()
            await self._connection.commit()
            return replace(claim, emission_started=True)
        except Exception:
            if self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def renew(
        self,
        claim: ProxyTurnClaim,
        *,
        lease_seconds: float = DEFAULT_PROXY_TURN_LEASE_SECONDS,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            run = await self.get_run(claim.pair_id)
            now = self._clock.now()
            self._require_current_owner(
                run,
                claim,
                allowed_states={"generating", "emission_started"},
                lease_boundary=now,
            )
            try:
                await TranscriptRebuildRepository(
                    self._connection,
                    self._clock,
                ).require_user_availability_snapshot(
                    claim.user_id,
                    UserAvailabilitySnapshot(
                        lifecycle_epoch=claim.lifecycle_epoch,
                        derivation_revision=claim.derivation_revision,
                    ),
                )
            except TranscriptRebuildInProgressError:
                try:
                    await self._fail_closed_nonterminal_run(
                        run=run,
                        error_code="proxy_source_revision_changed",
                        error_message=(
                            "Canonical memory sources changed after proxy turn admission"
                        ),
                        generating_conflict_code="proxy_source_revision_changed",
                        generating_conflict_message=(
                            "Canonical memory sources changed after this proxy turn was reserved; use new message IDs"
                        ),
                    )
                except ProxyTurnConflictError:
                    return False
            await self._require_claim_namespace_current(run=run, claim=claim)
            now = self._clock.now()
            self._require_current_owner(
                run,
                claim,
                allowed_states={"generating", "emission_started"},
                lease_boundary=now,
            )
            timestamp = now.isoformat()
            lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
            cursor = await self._connection.execute(
                """
                UPDATE proxy_turn_runs
                SET lease_expires_at = ?, updated_at = ?
                WHERE pair_id = ?
                  AND state IN ('generating', 'emission_started')
                  AND owner_token = ?
                  AND owner_fence = ?
                  AND lease_expires_at > ?
                """,
                (
                    lease_expires_at,
                    timestamp,
                    claim.pair_id,
                    claim.owner_token,
                    claim.owner_fence,
                    timestamp,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise self._stale_owner()
            await self._connection.commit()
            return True
        except (ProxyTurnConflictError, ProxyTurnStaleOwnerError):
            if self._connection.in_transaction:
                await self._connection.rollback()
            return False
        except Exception:
            if self._connection.in_transaction:
                await self._connection.rollback()
            raise

    async def claim_is_current(self, claim: ProxyTurnClaim) -> bool:
        started_transaction = not self._connection.in_transaction
        if started_transaction:
            await self._connection.execute("BEGIN")
        try:
            row = await self.get_run(claim.pair_id)
            current = self._clock.now()
            if (
                row is None
                or str(row.get("state")) not in {"generating", "emission_started"}
                or str(row.get("owner_token")) != claim.owner_token
                or int(row.get("owner_fence") or 0) != claim.owner_fence
                or not _timestamp_after(row.get("lease_expires_at"), current)
                or self._run_namespace_snapshot(row) != claim.namespace_snapshot
            ):
                return False
            await TranscriptRebuildRepository(
                self._connection,
                self._clock,
            ).require_user_availability_snapshot(
                claim.user_id,
                UserAvailabilitySnapshot(
                    lifecycle_epoch=claim.lifecycle_epoch,
                    derivation_revision=claim.derivation_revision,
                ),
            )
            namespace = await capture_conversation_namespace_snapshot(
                self._connection,
                self._clock,
                user_id=claim.user_id,
                conversation_id=claim.conversation_id,
            )
            return namespace == claim.namespace_snapshot and _timestamp_after(
                row.get("lease_expires_at"),
                self._clock.now(),
            )
        except (
            ProxyTurnConflictError,
            TranscriptRebuildInProgressError,
            ValueError,
        ):
            return False
        finally:
            if started_transaction and self._connection.in_transaction:
                await self._connection.rollback()

    async def abandon_pre_emission(
        self, claim: ProxyTurnClaim, error: Exception
    ) -> bool:
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE proxy_turn_runs
            SET lease_expires_at = ?,
                error_code = 'provider_pre_emission_failure',
                error_message = ?,
                updated_at = ?
            WHERE pair_id = ?
              AND state = 'generating'
              AND owner_token = ?
              AND owner_fence = ?
            """,
            (
                timestamp,
                str(error)[:500],
                timestamp,
                claim.pair_id,
                claim.owner_token,
                claim.owner_fence,
            ),
        )
        await self._connection.commit()
        return int(cursor.rowcount or 0) == 1

    async def mark_ambiguous_exposed(
        self,
        claim: ProxyTurnClaim,
        *,
        error_message: str,
    ) -> bool:
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE proxy_turn_runs
            SET state = 'ambiguous_exposed',
                owner_token = NULL,
                lease_expires_at = NULL,
                ambiguous_at = ?,
                error_code = 'stream_owner_lost',
                error_message = ?,
                updated_at = ?
            WHERE pair_id = ?
              AND state = 'emission_started'
              AND owner_token = ?
              AND owner_fence = ?
            """,
            (
                timestamp,
                error_message[:500],
                timestamp,
                claim.pair_id,
                claim.owner_token,
                claim.owner_fence,
            ),
        )
        await self._connection.commit()
        return int(cursor.rowcount or 0) == 1

    async def finalize(
        self,
        claim: ProxyTurnClaim,
        *,
        response: ProxyTurnResponseMessage,
        durable_jobs: Sequence[ProxyDurableJobInsert] | None = None,
        durable_job_builder: Callable[[], Awaitable[Sequence[ProxyDurableJobInsert]]]
        | None = None,
        turn_telemetry: ProxyTurnTelemetry,
        failpoint: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Atomically persist the response, telemetry, jobs, linkage, and completion."""

        if (durable_jobs is None) == (durable_job_builder is None):
            raise ValueError(
                "Provide exactly one of durable_jobs or durable_job_builder"
            )
        if claim.final_provider_fingerprint is None:
            raise ValueError(
                "A final provider fingerprint is required before completion"
            )
        allowed_state = "emission_started" if claim.emission_started else "generating"
        await self._connection.execute("BEGIN IMMEDIATE")
        try:
            await TranscriptRebuildRepository(
                self._connection,
                self._clock,
            ).require_user_availability_snapshot(
                claim.user_id,
                UserAvailabilitySnapshot(
                    lifecycle_epoch=claim.lifecycle_epoch,
                    derivation_revision=claim.derivation_revision,
                ),
            )
            if (
                await self._active_conversation(
                    claim.conversation_id,
                    claim.user_id,
                )
                is None
            ):
                raise ConversationNotFoundError("Conversation not found for user")
            run = await self.get_run(claim.pair_id)
            now = self._clock.now()
            self._require_current_owner(
                run,
                claim,
                allowed_states={allowed_state},
                lease_boundary=now,
            )
            await self._require_claim_namespace_current(run=run, claim=claim)
            if (
                run is None
                or str(run.get("final_provider_fingerprint"))
                != claim.final_provider_fingerprint
            ):
                raise self._stale_owner()
            request_row = await MessageRepository(
                self._connection,
                self._clock,
            ).get_message_for_idempotency(claim.request_message_id)
            if request_row is None:
                raise ProxyTurnConflictError(
                    "Proxy pair is missing its immutable input message",
                    code="proxy_pair_integrity_conflict",
                )
            resolved_durable_jobs = (
                list(await durable_job_builder())
                if durable_job_builder is not None
                else list(durable_jobs or ())
            )
            now = self._clock.now()
            run = await self.get_run(claim.pair_id)
            self._require_current_owner(
                run,
                claim,
                allowed_states={allowed_state},
                lease_boundary=now,
            )
            timestamp = now.isoformat()
            response_row = await self._insert_or_validate_response(
                claim=claim,
                request_row=request_row,
                response=response,
            )
            _call_failpoint(failpoint, "response_inserted")
            response_source_seq = int(response_row["seq"])
            # Inside the fence, and only now: the retrieval event's
            # response_message_id has a row to point at, and the turn's LLM
            # counters are final. A failure here aborts the whole terminal
            # commit rather than leaving an unmeasured turn behind.
            await self._write_turn_telemetry(claim, turn_telemetry)

            job_repository = JobRunRepository(self._connection, self._clock)
            job_ids: list[str] = []
            for durable_job in resolved_durable_jobs:
                await job_repository.create_durable_job(
                    stream_name=durable_job.stream_name,
                    target_backend=durable_job.target_backend,
                    envelope=durable_job.envelope,
                    source_token_estimate=durable_job.source_token_estimate,
                    size_bucket=durable_job.size_bucket,
                    queued_at=(
                        durable_job.envelope.created_at.isoformat()
                        if durable_job.envelope.created_at is not None
                        else timestamp
                    ),
                    metadata=durable_job.metadata,
                    user_persona_id=durable_job.user_persona_id,
                    platform_id=durable_job.platform_id,
                    character_id=durable_job.character_id,
                    incognito_snapshot=durable_job.incognito_snapshot,
                    remember_across_chats_snapshot=(
                        durable_job.remember_across_chats_snapshot
                    ),
                    remember_across_devices_snapshot=(
                        durable_job.remember_across_devices_snapshot
                    ),
                    temporary_snapshot=durable_job.temporary_snapshot,
                    purge_on_close_snapshot=durable_job.purge_on_close_snapshot,
                    policy_snapshot=durable_job.policy_snapshot,
                    commit=False,
                )
                job_ids.append(durable_job.envelope.job_id)
                _call_failpoint(
                    failpoint,
                    f"durable_job_inserted:{durable_job.envelope.job_id}",
                )

            now = self._clock.now()
            run = await self.get_run(claim.pair_id)
            self._require_current_owner(
                run,
                claim,
                allowed_states={allowed_state},
                lease_boundary=now,
            )
            timestamp = now.isoformat()

            cursor = await self._connection.execute(
                """
                UPDATE proxy_turn_runs
                SET response_linked_at = ?,
                    response_source_seq = COALESCE(response_source_seq, ?),
                    durable_job_ids_json = ?,
                    updated_at = ?
                WHERE pair_id = ?
                  AND state = ?
                  AND owner_token = ?
                  AND owner_fence = ?
                  AND lifecycle_epoch = ?
                  AND derivation_revision = ?
                  AND final_provider_fingerprint = ?
                  AND (response_source_seq IS NULL OR response_source_seq = ?)
                  AND lease_expires_at > ?
                """,
                (
                    timestamp,
                    response_source_seq,
                    canonical_json(job_ids),
                    timestamp,
                    claim.pair_id,
                    allowed_state,
                    claim.owner_token,
                    claim.owner_fence,
                    claim.lifecycle_epoch,
                    claim.derivation_revision,
                    claim.final_provider_fingerprint,
                    response_source_seq,
                    timestamp,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise self._stale_owner()
            _call_failpoint(failpoint, "response_linked")

            cursor = await self._connection.execute(
                """
                UPDATE proxy_turn_runs
                SET state = 'completed',
                    owner_token = NULL,
                    lease_expires_at = NULL,
                    completed_at = ?,
                    error_code = NULL,
                    error_message = NULL,
                    updated_at = ?
                WHERE pair_id = ?
                  AND state = ?
                  AND owner_token = ?
                  AND owner_fence = ?
                  AND lifecycle_epoch = ?
                  AND derivation_revision = ?
                  AND response_linked_at IS NOT NULL
                  AND final_provider_fingerprint = ?
                  AND lease_expires_at > ?
                """,
                (
                    timestamp,
                    timestamp,
                    claim.pair_id,
                    allowed_state,
                    claim.owner_token,
                    claim.owner_fence,
                    claim.lifecycle_epoch,
                    claim.derivation_revision,
                    claim.final_provider_fingerprint,
                    timestamp,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                raise self._stale_owner()
            _call_failpoint(failpoint, "run_completed")
            await self._connection.commit()
            return response_row
        except Exception:
            await self._connection.rollback()
            raise

    async def prune_terminal_diagnostics(self, *, before: datetime) -> int:
        cursor = await self._connection.execute(
            """
            UPDATE proxy_turn_runs
            SET owner_token = NULL,
                lease_expires_at = NULL,
                error_message = NULL,
                updated_at = updated_at
            WHERE state IN (
                'completed',
                'ambiguous_exposed',
                'final_fingerprint_conflict'
            )
              AND updated_at < ?
            """,
            (before.isoformat(),),
        )
        await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def get_run(self, pair_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            "SELECT * FROM proxy_turn_runs WHERE pair_id = ?",
            (pair_id,),
        )

    async def get_claims(self, pair_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM proxy_message_id_claims
            WHERE pair_id = ?
            ORDER BY pair_role ASC
            """,
            (pair_id,),
        )

    async def find_parent_tool_response(
        self,
        *,
        user_id: str,
        conversation_id: str,
        expected_calls: Sequence[Mapping[str, Any]],
        response_message_hint: str | None = None,
    ) -> str:
        """Resolve one stored parent response from its ordered call projection."""

        parameters: list[Any] = [conversation_id, user_id]
        hint_clause = ""
        if response_message_hint is not None:
            hint_clause = "AND message.id = ?"
            parameters.append(response_message_hint)
        rows = await self._fetch_all(
            f"""
            SELECT message.*
            FROM messages AS message
            JOIN conversations AS conversation
              ON conversation.id = message.conversation_id
            WHERE message.conversation_id = ?
              AND conversation.user_id = ?
              AND message.role = 'assistant'
              {hint_clause}
            ORDER BY message.seq DESC
            """,
            tuple(parameters),
        )
        expected = canonical_json(list(expected_calls))
        matches: list[str] = []
        for row in rows:
            metadata = row.get("metadata_json")
            transcript = (
                metadata.get(PROXY_TRANSCRIPT_METADATA_KEY)
                if isinstance(metadata, dict)
                else None
            )
            replay = transcript.get("replay") if isinstance(transcript, dict) else None
            replay_calls = (
                replay.get("tool_calls") if isinstance(replay, dict) else None
            )
            calls = (
                normalize_assistant_tool_calls(replay_calls)
                if isinstance(replay_calls, list)
                else None
            )
            if isinstance(calls, list) and canonical_json(calls) == expected:
                matches.append(str(row["id"]))
        if len(matches) != 1:
            raise ProxyTurnConflictError(
                "Tool continuation does not identify exactly one persisted parent response",
                code="proxy_tool_parent_conflict",
            )
        return matches[0]

    async def _active_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT conversation.*
            FROM conversations AS conversation
            JOIN users ON users.id = conversation.user_id
            WHERE conversation.id = ?
              AND conversation.user_id = ?
              AND conversation.status = 'active'
              AND users.deleted_at IS NULL
            """,
            (conversation_id, user_id),
        )

    async def _assert_ids_available_for_new_pair(
        self,
        request_message: ProxyTurnRequestMessage,
    ) -> None:
        for message_id in (
            request_message.message_id,
            request_message.response_message_id,
        ):
            claim = await self._fetch_one(
                "SELECT * FROM proxy_message_id_claims WHERE message_id = ?",
                (message_id,),
            )
            if claim is not None:
                raise self._message_id_conflict()
            message = await self._fetch_one(
                "SELECT id FROM messages WHERE id = ?",
                (message_id,),
            )
            if message is not None:
                raise self._message_id_conflict()

    async def _insert_claims(
        self,
        *,
        pair_id: str,
        request_message: ProxyTurnRequestMessage,
        claim_token: str,
        timestamp: str,
    ) -> None:
        await self._connection.executemany(
            """
            INSERT INTO proxy_message_id_claims(
                message_id,
                pair_id,
                pair_role,
                message_role,
                counterpart_message_id,
                user_id,
                conversation_id,
                claim_token,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    request_message.message_id,
                    pair_id,
                    "request",
                    request_message.role,
                    request_message.response_message_id,
                    request_message.user_id,
                    request_message.conversation_id,
                    claim_token,
                    timestamp,
                ),
                (
                    request_message.response_message_id,
                    pair_id,
                    "response",
                    "assistant",
                    request_message.message_id,
                    request_message.user_id,
                    request_message.conversation_id,
                    claim_token,
                    timestamp,
                ),
            ),
        )

    async def _validate_tool_parent_link(
        self,
        request_message: ProxyTurnRequestMessage,
    ) -> None:
        if request_message.role != "tool":
            return
        transcript = request_message.metadata.get(PROXY_TRANSCRIPT_METADATA_KEY)
        projection = (
            transcript.get("tool_projection") if isinstance(transcript, dict) else None
        )
        parent_message_id = (
            projection.get("parent_response_message_id")
            if isinstance(projection, dict)
            else None
        )
        results = projection.get("results") if isinstance(projection, dict) else None
        if not isinstance(parent_message_id, str) or not isinstance(results, list):
            raise ProxyTurnConflictError(
                "Tool-result input is missing its persisted parent linkage",
                code="proxy_tool_parent_conflict",
            )
        parent = await MessageRepository(
            self._connection,
            self._clock,
        ).get_message_for_idempotency(parent_message_id)
        parent_metadata = parent.get("metadata_json") if parent else None
        parent_transcript = (
            parent_metadata.get(PROXY_TRANSCRIPT_METADATA_KEY)
            if isinstance(parent_metadata, dict)
            else None
        )
        parent_projection = (
            parent_transcript.get("tool_projection")
            if isinstance(parent_transcript, dict)
            else None
        )
        calls = (
            parent_projection.get("calls")
            if isinstance(parent_projection, dict)
            else None
        )
        if (
            parent is None
            or str(parent.get("_conversation_user_id")) != request_message.user_id
            or str(parent.get("conversation_id")) != request_message.conversation_id
            or str(parent.get("role")) != "assistant"
            or not isinstance(calls, list)
        ):
            raise ProxyTurnConflictError(
                "Tool continuation parent is not a persisted assistant tool-call response",
                code="proxy_tool_parent_conflict",
            )
        calls_by_id = {
            str(call.get("id")): call
            for call in calls
            if isinstance(call, dict) and call.get("id") is not None
        }
        for result in results:
            if not isinstance(result, dict):
                raise ProxyTurnConflictError(
                    "Tool-result metadata is malformed",
                    code="proxy_tool_parent_conflict",
                )
            call = calls_by_id.get(str(result.get("tool_call_id")))
            if call is None or int(call.get("position", -1)) != int(
                result.get("call_position", -2)
            ):
                raise ProxyTurnConflictError(
                    "Tool result does not match its persisted parent call",
                    code="proxy_tool_parent_conflict",
                )

    async def _source_presence_id_for_request(
        self,
        *,
        conversation: dict[str, Any],
        request_role: Literal["user", "tool"],
    ) -> str:
        active_presence_id = conversation.get("active_presence_id")
        if not isinstance(active_presence_id, str) or not active_presence_id:
            raise ProxyTurnConflictError(
                "The proxy conversation has no active Presence",
                code="proxy_namespace_changed",
            )
        presences = PresenceRepository(self._connection, self._clock)
        active_row = await presences.get_presence(
            owner_user_id=str(conversation["user_id"]),
            presence_id=active_presence_id,
        )
        if active_row is None:
            raise ProxyTurnConflictError(
                "The proxy conversation active Presence is unavailable",
                code="proxy_namespace_changed",
            )
        active = presence_snapshot(active_row)
        if request_role == "tool":
            return active.presence_id
        human_row = await presences.resolve_human_owner_presence(
            owner_user_id=str(conversation["user_id"]),
            commit=False,
        )
        return presence_snapshot(human_row).presence_id

    async def _insert_request_message(
        self,
        *,
        conversation: dict[str, Any],
        request_message: ProxyTurnRequestMessage,
        claim_token: str,
        source_presence_id: str,
    ) -> dict[str, Any]:
        messages = MessageRepository(self._connection, self._clock)
        if request_message.source_seq is not None:
            existing = await messages.get_message_by_seq(
                request_message.conversation_id,
                request_message.user_id,
                request_message.source_seq,
            )
            if existing is not None:
                raise SourceSequenceConflictError(
                    "source_seq already exists for a different message in this conversation"
                )
        return await messages.create_message(
            message_id=request_message.message_id,
            conversation_id=request_message.conversation_id,
            role=request_message.role,
            seq=request_message.source_seq,
            text=request_message.text,
            metadata=request_message.metadata,
            occurred_at=request_message.occurred_at or self._timestamp(),
            active_presence_id=conversation.get("active_presence_id"),
            source_presence_id=source_presence_id,
            space_id=conversation.get("active_space_id"),
            active_mind_id=conversation.get("active_mind_id"),
            source_mind_id=conversation.get("active_mind_id"),
            active_embodiment_id=conversation.get("active_embodiment_id"),
            active_realm_id=conversation.get("active_realm_id"),
            proxy_claim_token=claim_token,
            proxy_pair_role="request",
            commit=False,
        )

    async def _validate_response_source_seq_for_new_pair(
        self,
        *,
        request_message: ProxyTurnRequestMessage,
        request_source_seq: int,
    ) -> None:
        response_source_seq = request_message.response_source_seq
        if response_source_seq is None:
            return
        if response_source_seq <= request_source_seq:
            raise SourceSequenceConflictError(
                "response_source_seq must follow the request message sequence"
            )
        occupied = await MessageRepository(
            self._connection,
            self._clock,
        ).get_message_by_seq(
            request_message.conversation_id,
            request_message.user_id,
            response_source_seq,
        )
        if occupied is not None:
            raise SourceSequenceConflictError(
                "response_source_seq already exists for a different message"
            )

    async def _validate_existing_pair(
        self,
        *,
        run: dict[str, Any],
        request_message: ProxyTurnRequestMessage,
        client_request_fingerprint: str,
    ) -> str:
        if (
            str(run["request_message_id"]) != request_message.message_id
            or str(run["response_message_id"]) != request_message.response_message_id
            or str(run["user_id"]) != request_message.user_id
            or str(run["conversation_id"]) != request_message.conversation_id
            or str(run["request_message_role"]) != request_message.role
        ):
            raise self._message_id_conflict()
        if (
            request_message.source_seq is not None
            and int(run["request_source_seq"]) != request_message.source_seq
        ):
            raise SourceSequenceConflictError(
                "source_seq does not match the reserved request message"
            )
        if request_message.response_source_seq is not None and (
            run.get("response_source_seq") is None
            or int(run["response_source_seq"]) != request_message.response_source_seq
        ):
            raise SourceSequenceConflictError(
                "response_source_seq does not match the reserved response message"
            )
        if str(run["client_request_fingerprint"]) != client_request_fingerprint:
            raise ProxyTurnConflictError(
                "The message ID pair was already used for a different proxy request",
                code="proxy_request_fingerprint_conflict",
            )
        claims = await self.get_claims(str(run["pair_id"]))
        if len(claims) != 2:
            raise ProxyTurnConflictError(
                "Proxy pair claims are incomplete",
                code="proxy_pair_integrity_conflict",
            )
        by_role = {str(claim["pair_role"]): claim for claim in claims}
        request_claim = by_role.get("request")
        response_claim = by_role.get("response")
        if (
            request_claim is None
            or response_claim is None
            or str(request_claim["message_id"]) != request_message.message_id
            or str(request_claim["counterpart_message_id"])
            != request_message.response_message_id
            or str(response_claim["message_id"]) != request_message.response_message_id
            or str(response_claim["counterpart_message_id"])
            != request_message.message_id
            or str(request_claim["claim_token"]) != str(response_claim["claim_token"])
        ):
            raise self._message_id_conflict()
        if str(run["state"]) == "completed":
            return str(request_claim["claim_token"])
        message = await MessageRepository(
            self._connection,
            self._clock,
        ).get_message_for_idempotency(request_message.message_id)
        if message is None:
            raise ProxyTurnConflictError(
                "Proxy pair is missing its immutable input message",
                code="proxy_pair_integrity_conflict",
            )
        metadata = message.get("metadata_json")
        if (
            str(message.get("_conversation_user_id")) != request_message.user_id
            or str(message.get("conversation_id")) != request_message.conversation_id
            or str(message.get("role")) != request_message.role
            or str(message.get("text")) != request_message.text
            or (
                request_message.source_seq is not None
                and int(message["seq"]) != request_message.source_seq
            )
            or idempotency_tool_projection(
                metadata if isinstance(metadata, dict) else None
            )
            != idempotency_tool_projection(request_message.metadata)
        ):
            raise MessageIdConflictError(
                "message_id already exists with different role, text, sequence, or tool metadata"
            )
        transcript = (
            metadata.get(PROXY_TRANSCRIPT_METADATA_KEY)
            if isinstance(metadata, dict)
            else None
        )
        if (
            not isinstance(transcript, dict)
            or transcript.get("pair_id") != run["pair_id"]
            or transcript.get("expected_response_message_id")
            != request_message.response_message_id
        ):
            raise ProxyTurnConflictError(
                "Proxy input message has incompatible reciprocal linkage",
                code="proxy_pair_integrity_conflict",
            )
        return str(request_claim["claim_token"])

    async def _completed_replay(self, run: dict[str, Any]) -> ProxyTurnReplay:
        response = await MessageRepository(
            self._connection,
            self._clock,
        ).get_message_for_idempotency(str(run["response_message_id"]))
        if response is None:
            raise ProxyTurnConflictError(
                "Completed proxy pair is missing its response",
                code="proxy_pair_integrity_conflict",
            )
        metadata = response.get("metadata_json")
        transcript = (
            metadata.get(PROXY_TRANSCRIPT_METADATA_KEY)
            if isinstance(metadata, dict)
            else None
        )
        replay = transcript.get("replay") if isinstance(transcript, dict) else None
        if (
            not isinstance(replay, dict)
            or transcript.get("pair_id") != run["pair_id"]
            or transcript.get("request_message_id") != run["request_message_id"]
        ):
            raise ProxyTurnConflictError(
                "Completed proxy response is not replayable",
                code="proxy_pair_integrity_conflict",
            )
        return ProxyTurnReplay(
            pair_id=str(run["pair_id"]),
            request_message_id=str(run["request_message_id"]),
            response_message_id=str(run["response_message_id"]),
            response_row=response,
            replay_envelope=dict(replay),
        )

    async def _write_turn_telemetry(
        self,
        claim: ProxyTurnClaim,
        turn_telemetry: ProxyTurnTelemetry,
    ) -> None:
        """Record the completed turn's telemetry on its retrieval event."""
        events = RetrievalEventRepository(self._connection, self._clock)
        telemetry = replace(
            turn_telemetry.telemetry,
            turn_to_event_write_wall_ms=(
                (perf_counter() - turn_telemetry.turn_started_at) * 1000.0
            ),
        )
        if turn_telemetry.retrieval_event_id is not None:
            await events.complete_turn_telemetry(
                turn_telemetry.retrieval_event_id,
                claim.user_id,
                response_message_id=claim.response_message_id,
                telemetry=telemetry,
                commit=False,
            )
            return
        await events.create_event(
            {
                **turn_telemetry.fallback_event,
                "response_message_id": claim.response_message_id,
            },
            telemetry=telemetry,
            commit=False,
        )

    async def _insert_or_validate_response(
        self,
        *,
        claim: ProxyTurnClaim,
        request_row: dict[str, Any],
        response: ProxyTurnResponseMessage,
    ) -> dict[str, Any]:
        messages = MessageRepository(self._connection, self._clock)
        existing = await messages.get_message_for_idempotency(claim.response_message_id)
        if existing is not None:
            metadata = existing.get("metadata_json")
            if (
                str(existing.get("_conversation_user_id")) != claim.user_id
                or str(existing.get("conversation_id")) != claim.conversation_id
                or str(existing.get("role")) != "assistant"
                or str(existing.get("text")) != response.text
                or (
                    response.source_seq is not None
                    and int(existing["seq"]) != response.source_seq
                )
                or idempotency_tool_projection(
                    metadata if isinstance(metadata, dict) else None
                )
                != idempotency_tool_projection(response.metadata)
            ):
                raise MessageIdConflictError(
                    "response_message_id already exists with incompatible transcript data"
                )
            existing_transcript = (
                metadata.get(PROXY_TRANSCRIPT_METADATA_KEY)
                if isinstance(metadata, dict)
                else None
            )
            expected_transcript = response.metadata.get(PROXY_TRANSCRIPT_METADATA_KEY)
            if canonical_json(existing_transcript) != canonical_json(
                expected_transcript
            ):
                raise MessageIdConflictError(
                    "response_message_id already exists with different replay metadata"
                )
            return existing
        if response.source_seq is not None:
            occupied = await messages.get_message_by_seq(
                claim.conversation_id,
                claim.user_id,
                response.source_seq,
            )
            if occupied is not None:
                raise SourceSequenceConflictError(
                    "response_source_seq already exists for a different message"
                )
        return await messages.create_message(
            message_id=claim.response_message_id,
            conversation_id=claim.conversation_id,
            role="assistant",
            seq=response.source_seq,
            text=response.text,
            metadata=response.metadata,
            occurred_at=response.occurred_at or self._timestamp(),
            active_presence_id=request_row.get("active_presence_id"),
            source_presence_id=request_row.get("active_presence_id"),
            space_id=request_row.get("space_id"),
            active_mind_id=request_row.get("active_mind_id"),
            source_mind_id=request_row.get("active_mind_id"),
            active_embodiment_id=request_row.get("active_embodiment_id"),
            active_realm_id=request_row.get("active_realm_id"),
            proxy_claim_token=claim.claim_token,
            proxy_pair_role="response",
            commit=False,
        )

    @staticmethod
    def _source_snapshot(run: dict[str, Any]) -> UserAvailabilitySnapshot:
        lifecycle_epoch = run.get("lifecycle_epoch")
        derivation_revision = run.get("derivation_revision")
        if lifecycle_epoch is None or derivation_revision is None:
            raise ProxyTurnConflictError(
                "Proxy turn has no canonical source snapshot; use new message IDs",
                code="proxy_source_snapshot_missing",
            )
        return UserAvailabilitySnapshot(
            lifecycle_epoch=str(lifecycle_epoch),
            derivation_revision=int(derivation_revision),
        )

    async def _current_namespace_snapshot(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> ConversationNamespaceSnapshot:
        snapshot = await capture_conversation_namespace_snapshot(
            self._connection,
            self._clock,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if snapshot is None:
            raise ConversationNotFoundError("Conversation not found for user")
        return snapshot

    @staticmethod
    def _run_namespace_snapshot(
        run: dict[str, Any],
    ) -> ConversationNamespaceSnapshot:
        payload = run.get("namespace_snapshot_json")
        try:
            return parse_conversation_namespace_snapshot(payload)
        except (TypeError, ValueError) as exc:
            raise ProxyTurnConflictError(
                "Proxy turn has no valid conversation namespace snapshot; use new message IDs",
                code="proxy_namespace_snapshot_invalid",
            ) from exc

    async def _require_run_namespace_current(
        self,
        *,
        run: dict[str, Any],
    ) -> ConversationNamespaceSnapshot:
        try:
            expected = self._run_namespace_snapshot(run)
        except ProxyTurnConflictError:
            await self._fail_closed_nonterminal_run(
                run=run,
                error_code="proxy_namespace_snapshot_invalid",
                error_message="Durable conversation namespace snapshot is invalid",
                generating_conflict_code="proxy_namespace_changed",
                generating_conflict_message=(
                    "The conversation namespace changed after this proxy turn was reserved; use new message IDs"
                ),
            )
            raise
        current = await capture_conversation_namespace_snapshot(
            self._connection,
            self._clock,
            user_id=str(run["user_id"]),
            conversation_id=str(run["conversation_id"]),
        )
        if current != expected:
            await self._fail_closed_nonterminal_run(
                run=run,
                error_code="proxy_namespace_changed",
                error_message=(
                    "Conversation namespace changed after proxy turn admission"
                ),
                generating_conflict_code="proxy_namespace_changed",
                generating_conflict_message=(
                    "The conversation namespace changed after this proxy turn was reserved; use new message IDs"
                ),
            )
        return expected

    async def _require_claim_namespace_current(
        self,
        *,
        run: dict[str, Any] | None,
        claim: ProxyTurnClaim,
    ) -> None:
        if run is None:
            raise self._stale_owner()
        stored = await self._require_run_namespace_current(run=run)
        if stored != claim.namespace_snapshot:
            raise self._stale_owner()

    async def _fail_closed_nonterminal_run(
        self,
        *,
        run: dict[str, Any],
        error_code: str,
        error_message: str,
        generating_conflict_code: str,
        generating_conflict_message: str,
    ) -> None:
        state = str(run.get("state") or "")
        timestamp = self._clock.now().isoformat()
        if state == "generating":
            cursor = await self._connection.execute(
                """
                UPDATE proxy_turn_runs
                SET state = 'final_fingerprint_conflict',
                    owner_token = NULL,
                    lease_expires_at = NULL,
                    error_code = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE pair_id = ?
                  AND state = 'generating'
                  AND owner_fence = ?
                """,
                (
                    error_code,
                    error_message,
                    timestamp,
                    str(run["pair_id"]),
                    int(run["owner_fence"]),
                ),
            )
            conflict = ProxyTurnConflictError(
                generating_conflict_message,
                code=generating_conflict_code,
            )
        elif state == "emission_started":
            cursor = await self._connection.execute(
                """
                UPDATE proxy_turn_runs
                SET state = 'ambiguous_exposed',
                    owner_token = NULL,
                    lease_expires_at = NULL,
                    ambiguous_at = COALESCE(ambiguous_at, ?),
                    error_code = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE pair_id = ?
                  AND state = 'emission_started'
                  AND owner_fence = ?
                """,
                (
                    timestamp,
                    error_code,
                    error_message,
                    timestamp,
                    str(run["pair_id"]),
                    int(run["owner_fence"]),
                ),
            )
            conflict = ProxyTurnConflictError(
                "The prior stream may have been partially exposed; retry with new message IDs",
                code="stream_retry_requires_new_ids",
            )
        else:
            raise self._stale_owner()
        if int(cursor.rowcount or 0) != 1:
            raise self._stale_owner()
        await self._connection.commit()
        raise conflict

    @staticmethod
    def _require_current_owner(
        run: dict[str, Any] | None,
        claim: ProxyTurnClaim,
        *,
        allowed_states: set[str],
        lease_boundary: datetime,
    ) -> None:
        if (
            run is None
            or str(run.get("state")) not in allowed_states
            or str(run.get("owner_token")) != claim.owner_token
            or int(run.get("owner_fence") or 0) != claim.owner_fence
            or not _timestamp_after(run.get("lease_expires_at"), lease_boundary)
        ):
            raise ProxyTurnRepository._stale_owner()

    @staticmethod
    def _message_id_conflict() -> ProxyTurnConflictError:
        return ProxyTurnConflictError(
            "A proxy message ID is already claimed in another namespace or pair role",
            code="proxy_message_id_claim_conflict",
        )

    @staticmethod
    def _stale_owner() -> ProxyTurnStaleOwnerError:
        return ProxyTurnStaleOwnerError(
            "Proxy generation ownership is no longer current",
            code="proxy_stale_owner",
        )


def _timestamp_after(value: Any, boundary: datetime) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed > boundary


def _call_failpoint(
    failpoint: Callable[[str], None] | None,
    name: str,
) -> None:
    if failpoint is not None:
        failpoint(name)
