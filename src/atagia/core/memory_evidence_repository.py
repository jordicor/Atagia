"""Persistence helpers for memory evidence packets."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, _encode_json
from atagia.models.schemas_memory import (
    MemoryEvidencePolarity,
    MemoryEvidenceSpeakerRelation,
    MemoryEvidenceSpanRole,
    MemoryEvidenceStatus,
    MemoryEvidenceSupportKind,
)


class MemoryEvidenceRepository(BaseRepository):
    """Stores source/trigger evidence packets for memory objects."""

    async def create_support_edge_with_spans(
        self,
        *,
        user_id: str,
        memory_id: str,
        support_kind: MemoryEvidenceSupportKind | str = MemoryEvidenceSupportKind.DIRECT,
        evidence_polarity: MemoryEvidencePolarity | str = MemoryEvidencePolarity.SUPPORTS,
        speaker_relation_to_subject: MemoryEvidenceSpeakerRelation | str = (
            MemoryEvidenceSpeakerRelation.UNKNOWN
        ),
        confidence: float = 0.5,
        confidence_details: dict[str, Any] | None = None,
        rationale: str | None = None,
        spans: list[dict[str, Any]] | None = None,
        status: MemoryEvidenceStatus | str = MemoryEvidenceStatus.ACTIVE,
        reuse_existing: bool = True,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Create or update one support edge and attach any missing spans."""

        resolved_support_kind = MemoryEvidenceSupportKind(support_kind)
        resolved_polarity = MemoryEvidencePolarity(evidence_polarity)
        resolved_speaker_relation = MemoryEvidenceSpeakerRelation(
            speaker_relation_to_subject
        )
        resolved_status = MemoryEvidenceStatus(status)
        normalized_confidence = max(0.0, min(1.0, float(confidence)))
        normalized_rationale = self._normalize_optional_text(rationale)

        await self._validate_memory_owner(user_id=user_id, memory_id=memory_id)

        edge = (
            await self._find_existing_edge(
                user_id=user_id,
                memory_id=memory_id,
                support_kind=resolved_support_kind,
                evidence_polarity=resolved_polarity,
                speaker_relation_to_subject=resolved_speaker_relation,
            )
            if reuse_existing
            else None
        )
        timestamp = self._timestamp()
        if edge is None:
            edge_id = generate_prefixed_id("mse")
            await self._connection.execute(
                """
                INSERT INTO memory_support_edges(
                    id,
                    user_id,
                    memory_id,
                    support_kind,
                    evidence_polarity,
                    speaker_relation_to_subject,
                    confidence,
                    confidence_json,
                    rationale,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge_id,
                    user_id,
                    memory_id,
                    resolved_support_kind.value,
                    resolved_polarity.value,
                    resolved_speaker_relation.value,
                    normalized_confidence,
                    _encode_json(confidence_details or {}),
                    normalized_rationale,
                    resolved_status.value,
                    timestamp,
                    timestamp,
                ),
            )
            edge = await self._get_edge(edge_id, user_id)
        else:
            edge_id = str(edge["id"])
            await self._connection.execute(
                """
                UPDATE memory_support_edges
                SET confidence = MAX(confidence, ?),
                    confidence_json = CASE
                        WHEN confidence_json = '{}' THEN ?
                        ELSE confidence_json
                    END,
                    rationale = COALESCE(rationale, ?),
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                """,
                (
                    normalized_confidence,
                    _encode_json(confidence_details or {}),
                    normalized_rationale,
                    timestamp,
                    edge_id,
                    user_id,
                ),
            )

        if edge is None:
            raise RuntimeError("Failed to create memory support edge")

        edge_id = str(edge["id"])
        for span in spans or []:
            await self._create_span_if_missing(
                user_id=user_id,
                memory_id=memory_id,
                support_edge_id=edge_id,
                span=span,
                timestamp=timestamp,
            )

        if commit:
            await self._connection.commit()
        return await self._packet_for_edge(user_id=user_id, edge_id=edge_id)

    async def list_packets_for_memory_ids(
        self,
        *,
        user_id: str,
        memory_ids: list[str],
        limit_per_memory: int = 2,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return prompt-ready support packets grouped by memory id."""

        normalized_memory_ids = self._stable_non_empty(memory_ids)
        if not normalized_memory_ids or limit_per_memory <= 0:
            return {}
        placeholders = ", ".join("?" for _ in normalized_memory_ids)
        edges = await self._fetch_all(
            f"""
            SELECT *
            FROM memory_support_edges
            WHERE user_id = ?
              AND memory_id IN ({placeholders})
              AND status = ?
            ORDER BY
                memory_id ASC,
                CASE support_kind
                    WHEN 'direct' THEN 0
                    WHEN 'contextual_direct' THEN 1
                    WHEN 'inferred' THEN 2
                    ELSE 3
                END ASC,
                CASE evidence_polarity
                    WHEN 'supports' THEN 0
                    WHEN 'qualifies' THEN 1
                    ELSE 2
                END ASC,
                confidence DESC,
                updated_at DESC,
                id ASC
            """,
            (user_id, *normalized_memory_ids, MemoryEvidenceStatus.ACTIVE.value),
        )
        selected_edges: list[dict[str, Any]] = []
        counts_by_memory: dict[str, int] = {}
        for edge in edges:
            memory_id = str(edge["memory_id"])
            count = counts_by_memory.get(memory_id, 0)
            if count >= limit_per_memory:
                continue
            counts_by_memory[memory_id] = count + 1
            selected_edges.append(edge)

        if not selected_edges:
            return {}

        edge_ids = [str(edge["id"]) for edge in selected_edges]
        spans_by_edge = await self._spans_by_edge(user_id=user_id, edge_ids=edge_ids)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for edge in selected_edges:
            packet = self._packet_from_edge(edge, spans_by_edge.get(str(edge["id"]), []))
            grouped.setdefault(str(edge["memory_id"]), []).append(packet)
        return grouped

    async def delete_for_memory_ids(
        self,
        *,
        user_id: str,
        memory_ids: list[str],
        commit: bool = True,
    ) -> int:
        normalized_memory_ids = self._stable_non_empty(memory_ids)
        if not normalized_memory_ids:
            return 0
        placeholders = ", ".join("?" for _ in normalized_memory_ids)
        cursor = await self._connection.execute(
            f"""
            DELETE FROM memory_support_edges
            WHERE user_id = ?
              AND memory_id IN ({placeholders})
            """,
            (user_id, *normalized_memory_ids),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def _validate_memory_owner(self, *, user_id: str, memory_id: str) -> None:
        row = await self._fetch_one(
            """
            SELECT id
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
            """,
            (memory_id, user_id),
        )
        if row is None:
            raise ValueError("Memory evidence requires an existing memory for the same user")

    async def _find_existing_edge(
        self,
        *,
        user_id: str,
        memory_id: str,
        support_kind: MemoryEvidenceSupportKind,
        evidence_polarity: MemoryEvidencePolarity,
        speaker_relation_to_subject: MemoryEvidenceSpeakerRelation,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_support_edges
            WHERE user_id = ?
              AND memory_id = ?
              AND support_kind = ?
              AND evidence_polarity = ?
              AND speaker_relation_to_subject = ?
              AND status = ?
            ORDER BY updated_at DESC, id ASC
            LIMIT 1
            """,
            (
                user_id,
                memory_id,
                support_kind.value,
                evidence_polarity.value,
                speaker_relation_to_subject.value,
                MemoryEvidenceStatus.ACTIVE.value,
            ),
        )

    async def _get_edge(self, edge_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            "SELECT * FROM memory_support_edges WHERE id = ? AND user_id = ?",
            (edge_id, user_id),
        )

    async def _packet_for_edge(
        self,
        *,
        user_id: str,
        edge_id: str,
    ) -> dict[str, Any]:
        edge = await self._get_edge(edge_id, user_id)
        if edge is None:
            raise RuntimeError(f"Failed to load memory support edge {edge_id}")
        spans_by_edge = await self._spans_by_edge(user_id=user_id, edge_ids=[edge_id])
        return self._packet_from_edge(edge, spans_by_edge.get(edge_id, []))

    async def _create_span_if_missing(
        self,
        *,
        user_id: str,
        memory_id: str,
        support_edge_id: str,
        span: dict[str, Any],
        timestamp: str,
    ) -> None:
        span_role = MemoryEvidenceSpanRole(span.get("span_role", "source"))
        quote_text = self._normalize_required_text(span.get("quote_text"))
        message_id = self._normalize_optional_text(span.get("message_id"))
        conversation_id = self._normalize_optional_text(span.get("conversation_id"))
        seq = self._optional_int(span.get("seq"))
        occurred_at = self._normalize_optional_text(span.get("occurred_at"))
        metadata = dict(span.get("metadata") or {})

        if message_id is not None:
            message = await self._message_for_span(
                user_id=user_id,
                message_id=message_id,
            )
            if message is None:
                raise ValueError("Memory evidence span message must belong to the same user")
            message_conversation_id = str(message["conversation_id"])
            if conversation_id is not None and conversation_id != message_conversation_id:
                raise ValueError("Memory evidence span conversation_id must match message conversation")
            conversation_id = message_conversation_id
            if seq is None and message.get("seq") is not None:
                seq = int(message["seq"])
            if occurred_at is None:
                occurred_at = self._normalize_optional_text(
                    message.get("occurred_at") or message.get("created_at")
                )
            metadata.setdefault("message_role", message.get("role"))

        existing = await self._fetch_one(
            """
            SELECT id
            FROM memory_evidence_spans
            WHERE user_id = ?
              AND support_edge_id = ?
              AND span_role = ?
              AND COALESCE(message_id, '') = COALESCE(?, '')
              AND quote_text = ?
            LIMIT 1
            """,
            (user_id, support_edge_id, span_role.value, message_id, quote_text),
        )
        if existing is not None:
            return

        await self._connection.execute(
            """
            INSERT INTO memory_evidence_spans(
                id,
                user_id,
                support_edge_id,
                memory_id,
                conversation_id,
                message_id,
                span_role,
                quote_text,
                char_start,
                char_end,
                seq,
                occurred_at,
                metadata_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generate_prefixed_id("mes"),
                user_id,
                support_edge_id,
                memory_id,
                conversation_id,
                message_id,
                span_role.value,
                quote_text,
                self._optional_int(span.get("char_start")),
                self._optional_int(span.get("char_end")),
                seq,
                occurred_at,
                _encode_json(metadata),
                timestamp,
                timestamp,
            ),
        )

    async def _message_for_span(
        self,
        *,
        user_id: str,
        message_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.id = ?
              AND c.user_id = ?
            """,
            (message_id, user_id),
        )

    async def _spans_by_edge(
        self,
        *,
        user_id: str,
        edge_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        normalized_edge_ids = self._stable_non_empty(edge_ids)
        if not normalized_edge_ids:
            return {}
        placeholders = ", ".join("?" for _ in normalized_edge_ids)
        rows = await self._fetch_all(
            f"""
            SELECT *
            FROM memory_evidence_spans
            WHERE user_id = ?
              AND support_edge_id IN ({placeholders})
            ORDER BY
                support_edge_id ASC,
                CASE span_role
                    WHEN 'source' THEN 0
                    WHEN 'trigger' THEN 1
                    WHEN 'qualifier' THEN 2
                    ELSE 3
                END ASC,
                COALESCE(seq, 999999999) ASC,
                created_at ASC,
                id ASC
            """,
            (user_id, *normalized_edge_ids),
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["support_edge_id"]), []).append(row)
        return grouped

    @staticmethod
    def _packet_from_edge(
        edge: dict[str, Any],
        spans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "id": edge["id"],
            "memory_id": edge["memory_id"],
            "support_kind": edge["support_kind"],
            "evidence_polarity": edge["evidence_polarity"],
            "speaker_relation_to_subject": edge["speaker_relation_to_subject"],
            "confidence": edge["confidence"],
            "confidence_details": edge.get("confidence_json") or {},
            "rationale": edge.get("rationale"),
            "status": edge["status"],
            "spans": spans,
        }

    @staticmethod
    def _stable_non_empty(values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @staticmethod
    def _normalize_required_text(value: Any) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("Memory evidence span quote_text must be non-empty")
        return normalized

    @staticmethod
    def _normalize_optional_text(value: Any) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split())
        return normalized or None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)
