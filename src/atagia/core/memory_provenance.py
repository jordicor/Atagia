"""Shared provenance writer for durable memory objects."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.memory_evidence_repository import MemoryEvidenceRepository
from atagia.core.repositories import MessageRepository
from atagia.core.source_references import SourceReference, source_sha256
from atagia.models.schemas_memory import (
    MemoryEvidencePolarity,
    MemoryEvidenceSpeakerRelation,
    MemoryEvidenceSpanRole,
    MemoryEvidenceSupportKind,
)


DEFAULT_MAX_SOURCE_SPANS = 2


class MemoryProvenanceWriter:
    """Creates structured evidence packets from source message provenance."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._message_repository = MessageRepository(connection, clock)
        self._evidence_repository = MemoryEvidenceRepository(connection, clock)

    async def create_packet_from_source_messages(
        self,
        *,
        user_id: str,
        memory_id: str,
        source_message_ids: list[str],
        writer_kind: str,
        support_kind: MemoryEvidenceSupportKind | str,
        evidence_polarity: MemoryEvidencePolarity | str = MemoryEvidencePolarity.SUPPORTS,
        speaker_relation_to_subject: MemoryEvidenceSpeakerRelation | str = (
            MemoryEvidenceSpeakerRelation.UNKNOWN
        ),
        confidence: float = 0.5,
        confidence_details: dict[str, Any] | None = None,
        rationale: str | None = None,
        source_reference_by_message_id: dict[str, SourceReference] | None = None,
        trigger_message_ids: list[str] | None = None,
        max_source_spans: int = DEFAULT_MAX_SOURCE_SPANS,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        """Create one support edge for a memory from real source messages."""

        normalized_source_ids = self._unique_strings(source_message_ids)
        if not normalized_source_ids:
            return None
        source_references = source_reference_by_message_id or {}
        if set(source_references) - set(normalized_source_ids):
            raise ValueError("Source references must belong to the packet's source messages")

        source_spans = await self._spans_for_message_ids(
            user_id=user_id,
            message_ids=self._representative_message_ids(
                normalized_source_ids,
                max_count=max_source_spans,
            ),
            span_role=MemoryEvidenceSpanRole.SOURCE,
            reference_by_message_id=source_references,
            metadata={"source": writer_kind},
        )
        if not source_spans:
            return None
        trigger_spans = await self._spans_for_message_ids(
            user_id=user_id,
            message_ids=self._unique_strings(trigger_message_ids or []),
            span_role=MemoryEvidenceSpanRole.TRIGGER,
            reference_by_message_id={},
            metadata={"source": f"{writer_kind}_trigger"},
        )
        spans = [*source_spans, *trigger_spans]

        resolved_details = dict(confidence_details or {})
        resolved_details.setdefault("writer_kind", writer_kind)
        resolved_details.setdefault("source_message_ids", normalized_source_ids)
        resolved_details.setdefault("source_message_count", len(normalized_source_ids))
        if len(source_spans) < len(normalized_source_ids):
            resolved_details.setdefault("source_span_policy", "first_last")

        return await self._evidence_repository.create_support_edge_with_spans(
            user_id=user_id,
            memory_id=memory_id,
            support_kind=support_kind,
            evidence_polarity=evidence_polarity,
            speaker_relation_to_subject=speaker_relation_to_subject,
            confidence=confidence,
            confidence_details=resolved_details,
            rationale=rationale,
            spans=spans,
            commit=commit,
        )

    async def _spans_for_message_ids(
        self,
        *,
        user_id: str,
        message_ids: list[str],
        span_role: MemoryEvidenceSpanRole,
        reference_by_message_id: dict[str, SourceReference],
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        for message_id in message_ids:
            message = await self._message_repository.get_message(message_id, user_id)
            reference = reference_by_message_id.get(message_id)
            if message is None:
                if reference is not None:
                    raise ValueError("Referenced source message does not belong to this user")
                continue
            message_text = str(message.get("text") or "")
            if reference is None:
                # Writers with no selected subrange explicitly cite the full message.
                if not message_text.strip():
                    continue
                reference = SourceReference(
                    char_start=len(message_text) - len(message_text.lstrip()),
                    char_end=len(message_text.rstrip()),
                    source_sha256=source_sha256(message_text),
                )
            quote = reference.quote(message_text)
            span_metadata = dict(metadata)
            span_metadata["source_reference"] = reference.model_dump(mode="json")
            spans.append(
                {
                    "span_role": span_role.value,
                    "message_id": message_id,
                    "conversation_id": message.get("conversation_id"),
                    "quote_text": quote,
                    "char_start": reference.char_start,
                    "char_end": reference.char_end,
                    "seq": message.get("seq"),
                    "occurred_at": message.get("occurred_at")
                    or message.get("created_at"),
                    "metadata": span_metadata,
                }
            )
        return spans

    @classmethod
    def _representative_message_ids(
        cls,
        message_ids: list[str],
        *,
        max_count: int,
    ) -> list[str]:
        normalized = cls._unique_strings(message_ids)
        if max_count <= 0 or len(normalized) <= max_count:
            return normalized
        if max_count == 1:
            return [normalized[0]]
        return cls._unique_strings([normalized[0], normalized[-1]])[:max_count]

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized
