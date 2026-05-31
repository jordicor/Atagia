"""Source-backed fact/facet projection repository."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository

_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


def fact_facet_value_key(value: str) -> str:
    """Return a mechanical key for comparing already-extracted values."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip().casefold()
    return normalized


def fact_facet_source_hash(quote_text: str) -> str:
    """Hash source span text without storing the text in diagnostic summaries."""

    return hashlib.sha256(str(quote_text or "").encode("utf-8")).hexdigest()


class MemoryFactFacetRepository(BaseRepository):
    """Stores indexed, source-backed fact/facet rows derived from memories."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        super().__init__(connection, clock)

    async def upsert_fact_facet(
        self,
        *,
        user_id: str,
        memory_id: str,
        source_span_id: str,
        source_message_id: str,
        subject_surface: str,
        facet_label: str,
        value_text: str,
        value_norm_key: str | None = None,
        conversation_id: str | None = None,
        subject_cluster_id: str | None = None,
        surface_class: str = "generic",
        value_type: str = "text",
        assertion_kind: str = "evidence",
        list_group_key: str | None = None,
        support_kind: str = "direct",
        observed_at: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        current_state: bool = True,
        supersedes_fact_id: str | None = None,
        temporal_phrase: str | None = None,
        temporal_anchor_at: str | None = None,
        resolved_interval_start: str | None = None,
        resolved_interval_end: str | None = None,
        temporal_granularity: str | None = None,
        temporal_resolution_type: str | None = None,
        temporal_confidence: float | None = None,
        language_code: str | None = None,
        confidence: float = 0.5,
        schema_version: int = 1,
        commit: bool = True,
    ) -> dict[str, Any]:
        memory = await self._fetch_one(
            """
            SELECT id
            FROM memory_objects
            WHERE user_id = ?
              AND id = ?
            """,
            (user_id, memory_id),
        )
        if memory is None:
            raise ValueError("memory_id must belong to user_id")

        source_span = await self._fetch_one(
            """
            SELECT *
            FROM memory_evidence_spans
            WHERE user_id = ?
              AND id = ?
              AND memory_id = ?
              AND span_role = 'source'
            """,
            (user_id, source_span_id, memory_id),
        )
        if source_span is None:
            raise ValueError("source_span_id must be a source span for the same user and memory")
        if str(source_span.get("message_id") or "") != source_message_id:
            raise ValueError("source_message_id must match source_span_id")

        normalized_subject = self._required_text(subject_surface, "subject_surface")
        normalized_facet = self._required_text(facet_label, "facet_label")
        normalized_value = self._required_text(value_text, "value_text")
        normalized_value_key = self._required_text(
            value_norm_key or fact_facet_value_key(normalized_value),
            "value_norm_key",
        )
        normalized_surface_class = self._required_text(surface_class, "surface_class")
        if normalized_surface_class not in {"structured", "generic"}:
            raise ValueError("surface_class must be one of: structured, generic")
        normalized_value_type = self._required_text(value_type, "value_type")
        normalized_assertion_kind = self._required_text(assertion_kind, "assertion_kind")
        normalized_support_kind = self._required_text(support_kind, "support_kind")
        normalized_confidence = max(0.0, min(1.0, float(confidence)))
        normalized_temporal_confidence = (
            None
            if temporal_confidence is None
            else max(0.0, min(1.0, float(temporal_confidence)))
        )
        resolved_conversation_id = conversation_id or source_span.get("conversation_id")
        resolved_observed_at = (
            observed_at
            or source_span.get("occurred_at")
            or self._timestamp()
        )
        updated_at = self._timestamp()
        source_hash = fact_facet_source_hash(str(source_span.get("quote_text") or ""))

        existing = await self._fetch_one(
            """
            SELECT *
            FROM memory_fact_facets
            WHERE user_id = ?
              AND memory_id = ?
              AND source_span_id = ?
              AND facet_label = ?
              AND value_norm_key = ?
              AND assertion_kind = ?
            """,
            (
                user_id,
                memory_id,
                source_span_id,
                normalized_facet,
                normalized_value_key,
                normalized_assertion_kind,
            ),
        )
        if existing is None:
            fact_id = generate_prefixed_id("mff")
            await self._connection.execute(
                """
                INSERT INTO memory_fact_facets(
                    id,
                    user_id,
                    conversation_id,
                    memory_id,
                    source_message_id,
                    source_span_id,
                    source_hash,
                    subject_surface,
                    subject_cluster_id,
                    surface_class,
                    facet_label,
                    value_text,
                    value_norm_key,
                    value_type,
                    assertion_kind,
                    list_group_key,
                    support_kind,
                    observed_at,
                    valid_from,
                    valid_to,
                    current_state,
                    supersedes_fact_id,
                    temporal_phrase,
                    temporal_anchor_at,
                    resolved_interval_start,
                    resolved_interval_end,
                    temporal_granularity,
                    temporal_resolution_type,
                    temporal_confidence,
                    language_code,
                    confidence,
                    schema_version,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_id,
                    user_id,
                    resolved_conversation_id,
                    memory_id,
                    source_message_id,
                    source_span_id,
                    source_hash,
                    normalized_subject,
                    self._optional_text(subject_cluster_id),
                    normalized_surface_class,
                    normalized_facet,
                    normalized_value,
                    normalized_value_key,
                    normalized_value_type,
                    normalized_assertion_kind,
                    self._optional_text(list_group_key),
                    normalized_support_kind,
                    str(resolved_observed_at),
                    self._optional_text(valid_from),
                    self._optional_text(valid_to),
                    int(current_state),
                    self._optional_text(supersedes_fact_id),
                    self._optional_text(temporal_phrase),
                    self._optional_text(temporal_anchor_at),
                    self._optional_text(resolved_interval_start),
                    self._optional_text(resolved_interval_end),
                    self._optional_text(temporal_granularity),
                    self._optional_text(temporal_resolution_type),
                    normalized_temporal_confidence,
                    self._optional_text(language_code),
                    normalized_confidence,
                    int(schema_version),
                    updated_at,
                    updated_at,
                ),
            )
        else:
            fact_id = str(existing["id"])
            await self._connection.execute(
                """
                UPDATE memory_fact_facets
                SET conversation_id = ?,
                    source_message_id = ?,
                    source_hash = ?,
                    subject_surface = ?,
                    subject_cluster_id = ?,
                    surface_class = ?,
                    value_text = ?,
                    value_type = ?,
                    list_group_key = ?,
                    support_kind = ?,
                    observed_at = ?,
                    valid_from = ?,
                    valid_to = ?,
                    current_state = ?,
                    supersedes_fact_id = ?,
                    temporal_phrase = ?,
                    temporal_anchor_at = ?,
                    resolved_interval_start = ?,
                    resolved_interval_end = ?,
                    temporal_granularity = ?,
                    temporal_resolution_type = ?,
                    temporal_confidence = ?,
                    language_code = ?,
                    confidence = ?,
                    schema_version = ?,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                """,
                (
                    resolved_conversation_id,
                    source_message_id,
                    source_hash,
                    normalized_subject,
                    self._optional_text(subject_cluster_id),
                    normalized_surface_class,
                    normalized_value,
                    normalized_value_type,
                    self._optional_text(list_group_key),
                    normalized_support_kind,
                    str(resolved_observed_at),
                    self._optional_text(valid_from),
                    self._optional_text(valid_to),
                    int(current_state),
                    self._optional_text(supersedes_fact_id),
                    self._optional_text(temporal_phrase),
                    self._optional_text(temporal_anchor_at),
                    self._optional_text(resolved_interval_start),
                    self._optional_text(resolved_interval_end),
                    self._optional_text(temporal_granularity),
                    self._optional_text(temporal_resolution_type),
                    normalized_temporal_confidence,
                    self._optional_text(language_code),
                    normalized_confidence,
                    int(schema_version),
                    updated_at,
                    fact_id,
                    user_id,
                ),
            )
        if commit:
            await self._connection.commit()
        refreshed = await self.get_fact_facet(user_id=user_id, fact_id=fact_id)
        if refreshed is None:
            raise RuntimeError(f"Failed to upsert memory fact/facet {fact_id}")
        return refreshed

    async def get_fact_facet(
        self,
        *,
        user_id: str,
        fact_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_fact_facets
            WHERE user_id = ?
              AND id = ?
            """,
            (user_id, fact_id),
        )

    async def list_for_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_fact_facets
            WHERE user_id = ?
              AND memory_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id, memory_id),
        )

    async def list_for_user(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if conversation_id is None:
            return await self._fetch_all(
                """
                SELECT *
                FROM memory_fact_facets
                WHERE user_id = ?
                ORDER BY created_at DESC, id ASC
                LIMIT ?
                """,
                (user_id, max(0, int(limit))),
            )
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_fact_facets
            WHERE user_id = ?
              AND conversation_id = ?
            ORDER BY created_at DESC, id ASC
            LIMIT ?
            """,
            (user_id, conversation_id, max(0, int(limit))),
        )

    async def health_counters(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        clause = "WHERE mff.user_id = ?"
        parameters: tuple[Any, ...] = (user_id,)
        if conversation_id is not None:
            clause += " AND mff.conversation_id = ?"
            parameters = (user_id, conversation_id)

        aggregate = await self._fetch_one(
            f"""
            SELECT
                COUNT(*) AS row_count,
                SUM(CASE WHEN span.id IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_source_spans,
                SUM(CASE WHEN mff.subject_cluster_id IS NULL THEN 1 ELSE 0 END) AS ambiguous_entity_rows,
                SUM(
                    CASE
                        WHEN mff.valid_from IS NOT NULL
                          OR mff.valid_to IS NOT NULL
                          OR mff.temporal_phrase IS NOT NULL
                          OR mff.resolved_interval_start IS NOT NULL
                          OR mff.resolved_interval_end IS NOT NULL
                          OR mff.temporal_confidence IS NOT NULL
                        THEN 1
                        ELSE 0
                    END
                ) AS temporal_rows
            FROM memory_fact_facets AS mff
            LEFT JOIN memory_evidence_spans AS span
              ON span.id = mff.source_span_id
             AND span.user_id = mff.user_id
            {clause}
            """,
            parameters,
        )
        grouped_rows = await self._fetch_all(
            f"""
            SELECT
                mff.conversation_id,
                COUNT(*) AS count
            FROM memory_fact_facets AS mff
            {clause}
            GROUP BY mff.conversation_id
            ORDER BY mff.conversation_id ASC
            """,
            parameters,
        )
        stale_rows = await self._fetch_all(
            f"""
            SELECT
                mff.id,
                mff.source_hash,
                span.quote_text AS source_quote_text
            FROM memory_fact_facets AS mff
            LEFT JOIN memory_evidence_spans AS span
              ON span.id = mff.source_span_id
             AND span.user_id = mff.user_id
            {clause}
            ORDER BY mff.id ASC
            """,
            parameters,
        )

        stale_source_hash_rows = 0
        for row in stale_rows:
            quote_text = row.get("source_quote_text")
            if quote_text is not None and fact_facet_source_hash(str(quote_text)) != row.get("source_hash"):
                stale_source_hash_rows += 1

        return {
            "row_count": int(aggregate.get("row_count") or 0) if aggregate else 0,
            "rows_by_conversation": {
                row.get("conversation_id"): int(row.get("count") or 0)
                for row in grouped_rows
            },
            "rows_with_source_spans": int(aggregate.get("rows_with_source_spans") or 0) if aggregate else 0,
            "temporal_rows": int(aggregate.get("temporal_rows") or 0) if aggregate else 0,
            "ambiguous_entity_rows": int(aggregate.get("ambiguous_entity_rows") or 0) if aggregate else 0,
            "stale_source_hash_rows": stale_source_hash_rows,
        }

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        normalized = _WHITESPACE_PATTERN.sub(" ", str(value or "")).strip()
        if not normalized:
            raise ValueError(f"{field_name} must be non-empty")
        return normalized

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        normalized = _WHITESPACE_PATTERN.sub(" ", str(value)).strip()
        return normalized or None
