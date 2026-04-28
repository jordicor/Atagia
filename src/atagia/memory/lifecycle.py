"""Memory lifecycle management for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import Any

import aiosqlite
from pydantic import BaseModel, ConfigDict

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.consent_repository import MemoryConsentProfileRepository, PendingMemoryConfirmationRepository
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.memory.consent_confirmation import PENDING_USER_CONFIRMATION_TTL_DAYS
from atagia.models.schemas_memory import MemoryCategory, MemoryObjectType, MemoryScope, MemoryStatus
from atagia.services.embeddings import EmbeddingIndex, NoneBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _LifecycleMemoryRef:
    memory_id: str
    user_id: str


class LifecycleCycleResult(BaseModel):
    """Outcome counters for one lifecycle execution."""

    model_config = ConfigDict(extra="forbid")

    decayed_count: int = 0
    archived_count: int = 0
    deleted_count: int = 0
    declined_count: int = 0
    skipped_evidence_count: int = 0


class MemoryLifecycleManager:
    """Applies archive-first lifecycle rules to memory objects."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        settings: Settings | None = None,
        embedding_index: EmbeddingIndex | None = None,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._embedding_index = embedding_index or NoneBackend()
        self._settings = settings or Settings.from_env()
        self._contract_repository = ContractDimensionRepository(connection, clock)
        self._consent_profile_repository = MemoryConsentProfileRepository(connection, clock)
        self._pending_confirmation_repository = PendingMemoryConfirmationRepository(connection, clock)
        self._affected_user_ids: set[str] = set()

    @property
    def affected_user_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._affected_user_ids))

    async def run_cycle(self, dry_run: bool = False) -> LifecycleCycleResult:
        """Run one lifecycle maintenance cycle."""
        now = self._clock.now()
        timestamp = now.isoformat()
        decay_cutoff = (now - timedelta(days=self._settings.lifecycle_decay_days)).isoformat()
        ephemeral_cutoff = (
            now - timedelta(hours=self._settings.lifecycle_ephemeral_ttl_hours)
        ).isoformat()
        review_cutoff = (now - timedelta(days=self._settings.lifecycle_review_ttl_days)).isoformat()
        pending_cutoff = (now - timedelta(days=PENDING_USER_CONFIRMATION_TTL_DAYS)).isoformat()
        result = LifecycleCycleResult()
        self._affected_user_ids = set()

        await self._connection.execute("BEGIN")
        try:
            decayed_refs = await self._decay_vitality(decay_cutoff, timestamp)
            result.decayed_count = len(decayed_refs)
            declined_refs = await self._decline_expired_pending_confirmations(pending_cutoff, timestamp)
            result.declined_count = len(declined_refs)
            expired_state_refs = await self._archive_expired_state_snapshots(timestamp)
            archived_refs = expired_state_refs + await self._archive_low_value_memories(timestamp)
            archived_ids = [ref.memory_id for ref in archived_refs]
            result.archived_count = len(archived_ids)
            result.skipped_evidence_count = await self._count_preserved_evidence()
            ephemeral_refs = await self._delete_expired_ephemeral(ephemeral_cutoff)
            ephemeral_ids = [ref.memory_id for ref in ephemeral_refs]
            review_required_refs = await self._delete_expired_review_required(review_cutoff)
            review_required_ids = [ref.memory_id for ref in review_required_refs]
            affected_user_ids = {
                ref.user_id
                for ref in decayed_refs + declined_refs + archived_refs + ephemeral_refs + review_required_refs
            }
            projection_keys = await self._contract_repository.list_projection_keys_for_sources(
                archived_ids + ephemeral_ids + review_required_ids
            )
            await self._delete_memory_ids(ephemeral_ids)
            await self._delete_memory_ids(review_required_ids)
            deleted_memory_ids = ephemeral_ids + review_required_ids
            result.deleted_count = len(ephemeral_ids) + len(review_required_ids)
            await self._refresh_contract_dimensions(projection_keys)
            await self._cleanup_orphaned_contract_dimensions()

            if dry_run:
                await self._connection.rollback()
            else:
                await self._connection.commit()
                await self._delete_embeddings(
                    archived_ids + deleted_memory_ids + [ref.memory_id for ref in declined_refs]
                )
                self._affected_user_ids = affected_user_ids
            return result
        except BaseException:
            self._affected_user_ids = set()
            await self._connection.rollback()
            raise

    async def _decay_vitality(self, decay_cutoff: str, timestamp: str) -> list[_LifecycleMemoryRef]:
        cursor = await self._connection.execute(
            """
            SELECT id, user_id, vitality
            FROM memory_objects
            WHERE status = ?
              AND updated_at < ?
            """,
            (MemoryStatus.ACTIVE.value, decay_cutoff),
        )
        rows = await cursor.fetchall()
        updates = [
            (
                float(row["vitality"]) * self._settings.lifecycle_decay_rate,
                timestamp,
                row["id"],
            )
            for row in rows
        ]
        if updates:
            await self._connection.executemany(
                """
                UPDATE memory_objects
                SET vitality = ?, updated_at = ?
                WHERE id = ?
                """,
                updates,
            )
        return [
            _LifecycleMemoryRef(memory_id=str(row["id"]), user_id=str(row["user_id"]))
            for row in rows
        ]

    async def _archive_low_value_memories(self, timestamp: str) -> list[_LifecycleMemoryRef]:
        cursor = await self._connection.execute(
            """
            SELECT id, user_id, canonical_text, payload_json
            FROM memory_objects
            WHERE status = ?
              AND object_type NOT IN (?, ?, ?)
              AND vitality < ?
              AND confidence < ?
            """,
            (
                MemoryStatus.ACTIVE.value,
                MemoryObjectType.EVIDENCE.value,
                MemoryObjectType.BELIEF.value,
                MemoryObjectType.STATE_SNAPSHOT.value,
                self._settings.lifecycle_archive_vitality,
                self._settings.lifecycle_archive_confidence,
            ),
        )
        rows = await cursor.fetchall()
        updates = [
            (
                MemoryStatus.ARCHIVED.value,
                json_utils.dumps(
                    self._archived_payload(row["payload_json"], row["canonical_text"]),
                    sort_keys=True,
                ),
                timestamp,
                row["id"],
            )
            for row in rows
        ]
        if updates:
            await self._connection.executemany(
                """
                UPDATE memory_objects
                SET status = ?, payload_json = ?, updated_at = ?
                WHERE id = ?
                """,
                updates,
            )
        return [
            _LifecycleMemoryRef(
                memory_id=str(row["id"]),
                user_id=str(row["user_id"]),
            )
            for row in rows
        ]

    async def _archive_expired_state_snapshots(self, timestamp: str) -> list[_LifecycleMemoryRef]:
        cursor = await self._connection.execute(
            """
            SELECT id, user_id, canonical_text, payload_json, valid_from, valid_to, temporal_type
            FROM memory_objects
            WHERE status = ?
              AND object_type = ?
            """,
            (
                MemoryStatus.ACTIVE.value,
                MemoryObjectType.STATE_SNAPSHOT.value,
            ),
        )
        reference = self._parse_temporal_datetime(timestamp, self._clock.now())
        rows = [
            row
            for row in await cursor.fetchall()
            if self._is_expired_state_snapshot(dict(row), reference)
        ]
        updates = [
            (
                MemoryStatus.ARCHIVED.value,
                json_utils.dumps(
                    self._archived_payload(row["payload_json"], row["canonical_text"]),
                    sort_keys=True,
                ),
                timestamp,
                row["id"],
            )
            for row in rows
        ]
        if updates:
            await self._connection.executemany(
                """
                UPDATE memory_objects
                SET status = ?, payload_json = ?, updated_at = ?
                WHERE id = ?
                """,
                updates,
            )
        return [
            _LifecycleMemoryRef(
                memory_id=str(row["id"]),
                user_id=str(row["user_id"]),
            )
            for row in rows
        ]

    def _is_expired_state_snapshot(self, row: dict[str, Any], reference: datetime) -> bool:
        temporal_type = str(row.get("temporal_type", "unknown"))
        if temporal_type == "ephemeral":
            valid_from = self._parse_temporal_datetime(row.get("valid_from"), reference)
            if valid_from is None:
                return True
            valid_to = self._parse_temporal_datetime(row.get("valid_to"), reference)
            effective_end = valid_to or (
                valid_from + timedelta(hours=self._settings.ephemeral_scoring_hours)
            )
            return effective_end < reference

        valid_to = self._parse_temporal_datetime(row.get("valid_to"), reference)
        return valid_to is not None and valid_to < reference

    @staticmethod
    def _parse_temporal_datetime(value: Any, reference: datetime) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None and reference.tzinfo is not None:
            parsed = parsed.replace(tzinfo=reference.tzinfo)
        return parsed

    async def _count_preserved_evidence(self) -> int:
        cursor = await self._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM memory_objects
            WHERE status = ?
              AND object_type = ?
              AND vitality < ?
              AND confidence < ?
            """,
            (
                MemoryStatus.ACTIVE.value,
                MemoryObjectType.EVIDENCE.value,
                self._settings.lifecycle_archive_vitality,
                self._settings.lifecycle_archive_confidence,
            ),
        )
        row = await cursor.fetchone()
        return int(row["count"])

    async def _decline_expired_pending_confirmations(
        self,
        cutoff: str,
        timestamp: str,
    ) -> list[_LifecycleMemoryRef]:
        cursor = await self._connection.execute(
            """
            SELECT id, user_id, memory_category, payload_json
            FROM memory_objects
            WHERE status = ?
              AND created_at < ?
            """,
            (
                MemoryStatus.PENDING_USER_CONFIRMATION.value,
                cutoff,
            ),
        )
        rows = await cursor.fetchall()
        updates = [
            (
                MemoryStatus.DECLINED.value,
                json_utils.dumps(
                    self._declined_payload(row["payload_json"], decline_reason="ttl_expired"),
                    sort_keys=True,
                ),
                timestamp,
                row["id"],
            )
            for row in rows
        ]
        if updates:
            await self._connection.executemany(
                """
                UPDATE memory_objects
                SET status = ?, payload_json = ?, updated_at = ?
                WHERE id = ?
                """,
                updates,
            )
            memory_ids_by_user: dict[str, list[str]] = {}
            decline_counts: dict[tuple[str, str], int] = {}
            for row in rows:
                user_id = str(row["user_id"])
                memory_ids_by_user.setdefault(user_id, []).append(str(row["id"]))
                category_key = (user_id, str(row["memory_category"]))
                decline_counts[category_key] = decline_counts.get(category_key, 0) + 1
            for user_id, memory_ids in memory_ids_by_user.items():
                await self._pending_confirmation_repository.clear_markers(
                    user_id,
                    memory_ids,
                    commit=False,
                )
            for (user_id, category_str), count in decline_counts.items():
                category = MemoryCategory(category_str)
                profile = await self._consent_profile_repository.get_profile(user_id, category)
                current_declined = int(profile.get("declined_count", 0)) if profile else 0
                await self._consent_profile_repository.upsert_profile(
                    user_id=user_id,
                    category=category,
                    confirmed_count=int(profile.get("confirmed_count", 0)) if profile else 0,
                    declined_count=current_declined + count,
                    last_confirmed_at=profile.get("last_confirmed_at") if profile else None,
                    last_declined_at=timestamp,
                    commit=False,
                )
        return [
            _LifecycleMemoryRef(memory_id=str(row["id"]), user_id=str(row["user_id"]))
            for row in rows
        ]

    async def _delete_expired_ephemeral(self, cutoff: str) -> list[_LifecycleMemoryRef]:
        return await self._select_memory_refs(
            """
            SELECT id, user_id
            FROM memory_objects
            WHERE scope = ?
              AND object_type != ?
              AND status IN (?, ?, ?, ?)
              AND created_at < ?
            """,
            (
                MemoryScope.EPHEMERAL_SESSION.value,
                MemoryObjectType.EVIDENCE.value,
                MemoryStatus.ACTIVE.value,
                MemoryStatus.SUPERSEDED.value,
                MemoryStatus.ARCHIVED.value,
                MemoryStatus.REVIEW_REQUIRED.value,
                cutoff,
            ),
        )

    async def _delete_expired_review_required(self, cutoff: str) -> list[_LifecycleMemoryRef]:
        return await self._select_memory_refs(
            """
            SELECT id, user_id
            FROM memory_objects
            WHERE status = ?
              AND object_type != ?
              AND created_at < ?
            """,
            (
                MemoryStatus.REVIEW_REQUIRED.value,
                MemoryObjectType.EVIDENCE.value,
                cutoff,
            ),
        )

    async def _cleanup_orphaned_contract_dimensions(self) -> None:
        await self._connection.execute(
            """
            DELETE FROM contract_dimensions_current
            WHERE source_memory_id IN (
                SELECT mo.id
                FROM memory_objects AS mo
                WHERE mo.status IN (?, ?)
            )
               OR NOT EXISTS (
                    SELECT 1
                    FROM memory_objects AS mo
                    WHERE mo.id = contract_dimensions_current.source_memory_id
                )
            """,
            (MemoryStatus.ARCHIVED.value, MemoryStatus.DELETED.value),
        )

    async def _refresh_contract_dimensions(self, projection_keys: list[dict[str, Any]]) -> None:
        seen: set[tuple[str, str | None, str | None, str | None, str, str]] = set()
        for key in projection_keys:
            dedupe_key = (
                str(key["user_id"]),
                key["assistant_mode_id"],
                key["workspace_id"],
                key["conversation_id"],
                str(key["scope"]),
                str(key["dimension_name"]),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            await self._contract_repository.reproject_best_remaining(
                user_id=str(key["user_id"]),
                assistant_mode_id=key["assistant_mode_id"],
                workspace_id=key["workspace_id"],
                conversation_id=key["conversation_id"],
                scope=MemoryScope(str(key["scope"])),
                dimension_name=str(key["dimension_name"]),
                commit=False,
            )

    async def _select_memory_refs(
        self,
        query: str,
        parameters: tuple[Any, ...],
    ) -> list[_LifecycleMemoryRef]:
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        return [
            _LifecycleMemoryRef(
                memory_id=str(row["id"]),
                user_id=str(row["user_id"]),
            )
            for row in rows
        ]

    async def _delete_memory_ids(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        placeholders = ", ".join("?" for _ in memory_ids)
        await self._connection.execute(
            f"DELETE FROM memory_objects WHERE id IN ({placeholders})",
            tuple(memory_ids),
        )

    async def _delete_embeddings(self, memory_ids: list[str]) -> None:
        for memory_id in memory_ids:
            try:
                await self._embedding_index.delete(memory_id)
            except Exception:
                logger.warning("Embedding cleanup failed for memory_id=%s", memory_id, exc_info=True)

    @staticmethod
    def _archived_payload(raw_payload: str | None, canonical_text: str) -> dict[str, Any]:
        payload: dict[str, Any]
        if isinstance(raw_payload, str) and raw_payload:
            payload = json_utils.loads(raw_payload)
        else:
            payload = {}
        payload.setdefault("archived_summary", canonical_text)
        return payload

    @staticmethod
    def _declined_payload(raw_payload: str | None, *, decline_reason: str) -> dict[str, Any]:
        payload: dict[str, Any]
        if isinstance(raw_payload, str) and raw_payload:
            payload = json_utils.loads(raw_payload)
        else:
            payload = {}
        payload["decline_reason"] = decline_reason
        return payload
