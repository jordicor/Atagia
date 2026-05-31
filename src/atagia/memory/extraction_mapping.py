"""Map the lean model-facing extraction contract into the rich result.

The extractor asks the LLM for :class:`LeanExtractionResult` (a small, flat
schema mid-tier models can satisfy reliably) and then expands each candidate
into the rich ``Extracted*`` objects the persistence path already consumes.
Everything the model no longer produces receives a server-side default that was
verified field-by-field against the downstream consumers (F1.1 inventory).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atagia.models.schemas_memory import (
    ExtractedBelief,
    ExtractedMemoryBase,
    ExtractedContractSignal,
    ExtractedEvidence,
    ExtractedStateUpdate,
    ExtractionResult,
    LeanExtractionCandidate,
    LeanExtractionResult,
    MemoryCategory,
    MemoryEvidencePolarity,
    MemoryEvidenceSpeakerRelation,
    MemoryScope,
    MemorySensitivity,
    MemoryObjectType,
    MemorySourceKind,
)

# Temporal confidence the mapper assigns when the lean candidate carries a usable
# temporal status. It must clear ``TEMPORAL_CONFIDENCE_THRESHOLD`` (0.6) so the
# extractor's persistence gate keeps the resolved bounds, without claiming full
# certainty the model never expressed.
_MAPPED_TEMPORAL_CONFIDENCE_PRESENT = 0.8
_MAPPED_TEMPORAL_CONFIDENCE_ABSENT = 0.0
FACT_FACET_PROJECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SourceBackedFactFacetProjection:
    """One source-backed fact/facet row derived from structured extraction data."""

    memory_id: str
    source_span_id: str
    source_message_id: str
    conversation_id: str | None
    subject_surface: str
    subject_cluster_id: str | None
    surface_class: str
    facet_label: str
    value_text: str
    value_type: str
    assertion_kind: str
    list_group_key: str | None
    support_kind: str
    observed_at: str | None
    valid_from: str | None
    valid_to: str | None
    current_state: bool
    supersedes_fact_id: str | None
    temporal_phrase: str | None
    temporal_anchor_at: str | None
    resolved_interval_start: str | None
    resolved_interval_end: str | None
    temporal_granularity: str | None
    temporal_resolution_type: str | None
    temporal_confidence: float | None
    language_code: str | None
    confidence: float
    schema_version: int = FACT_FACET_PROJECTION_SCHEMA_VERSION


def _has_usable_temporal_status(candidate: LeanExtractionCandidate) -> bool:
    status = candidate.temporal_status
    if status is None:
        return False
    if status.type != "unknown":
        return True
    return bool(status.valid_from_iso or status.valid_to_iso)


def _common_fields(candidate: LeanExtractionCandidate) -> dict[str, object]:
    """Build the shared rich fields with approved server-side defaults."""

    has_temporal = _has_usable_temporal_status(candidate)
    if has_temporal:
        status = candidate.temporal_status
        temporal_type = status.type
        valid_from_iso = status.valid_from_iso
        valid_to_iso = status.valid_to_iso
        temporal_confidence = _MAPPED_TEMPORAL_CONFIDENCE_PRESENT
    else:
        temporal_type = "unknown"
        valid_from_iso = None
        valid_to_iso = None
        temporal_confidence = _MAPPED_TEMPORAL_CONFIDENCE_ABSENT

    return {
        "canonical_text": candidate.canonical_text,
        "index_text": candidate.index_text,
        "scope": MemoryScope(candidate.subject_scope),
        "confidence": candidate.confidence,
        "source_kind": MemorySourceKind.EXTRACTED,
        "support_kind": candidate.support_kind,
        "evidence_polarity": MemoryEvidencePolarity.SUPPORTS,
        "speaker_relation_to_subject": MemoryEvidenceSpeakerRelation.UNKNOWN,
        "source_quote": candidate.source_span,
        "trigger_message_ids": [],
        "trigger_quote": None,
        "support_rationale": None,
        "confidence_details": {},
        "privacy_level": 0,
        "sensitivity": MemorySensitivity.UNKNOWN,
        "themes": [],
        "auto_expires": False,
        "platform_locked": False,
        "memory_category": MemoryCategory.UNKNOWN,
        "informational_mention": None,
        "preserve_verbatim": candidate.preserve_verbatim,
        "subject_presence_ids": [],
        "payload": {},
        "temporal_type": temporal_type,
        "valid_from_iso": valid_from_iso,
        "valid_to_iso": valid_to_iso,
        "temporal_confidence": temporal_confidence,
        "language_codes": list(candidate.language_codes),
    }


def lean_result_to_extraction_result(lean: LeanExtractionResult) -> ExtractionResult:
    """Expand a lean extraction result into the rich :class:`ExtractionResult`.

    Each candidate is routed to its rich bucket by ``kind``. The produced objects
    satisfy the rich validators directly; the legacy normalization shims on the
    rich models are never exercised.
    """

    evidences: list[ExtractedEvidence] = []
    beliefs: list[ExtractedBelief] = []
    contract_signals: list[ExtractedContractSignal] = []
    state_updates: list[ExtractedStateUpdate] = []

    for candidate in lean.candidates:
        fields = _common_fields(candidate)
        if candidate.kind == "belief":
            beliefs.append(
                ExtractedBelief(
                    **fields,
                    claim_key=candidate.claim_key,
                    claim_value=candidate.claim_value,
                )
            )
        elif candidate.kind == "contract_signal":
            contract_signals.append(ExtractedContractSignal(**fields))
        elif candidate.kind == "state_update":
            state_updates.append(ExtractedStateUpdate(**fields))
        else:
            evidences.append(ExtractedEvidence(**fields))

    has_items = bool(evidences or beliefs or contract_signals or state_updates)
    return ExtractionResult(
        evidences=evidences,
        beliefs=beliefs,
        contract_signals=contract_signals,
        state_updates=state_updates,
        nothing_durable=lean.nothing_durable and not has_items,
    )


def source_backed_fact_facet_projection(
    *,
    item: ExtractedMemoryBase,
    object_type: MemoryObjectType,
    memory_row: dict[str, Any],
    evidence_packet: dict[str, Any] | None,
) -> SourceBackedFactFacetProjection | None:
    """Map a persisted memory plus source packet to one fact/facet projection.

    This function intentionally consumes only already-structured extraction
    fields and source packet metadata. It does not infer entities, facets, or
    list members from natural-language text.
    """

    source_span = _first_source_span(evidence_packet)
    if source_span is None:
        return None
    source_span_id = _optional_text(source_span.get("id"))
    source_message_id = _optional_text(source_span.get("message_id"))
    if source_span_id is None or source_message_id is None:
        raise ValueError("source-backed fact facet projection requires source span id and message id")

    memory_id = _optional_text(memory_row.get("id"))
    if memory_id is None:
        raise ValueError("source-backed fact facet projection requires memory id")
    memory_payload = _dict_or_empty(memory_row.get("payload_json"))
    item_payload = _dict_or_empty(item.payload)
    payload = {**memory_payload, **item_payload}

    facet_label, value_text, surface_class = _facet_value_for_item(
        item=item,
        object_type=object_type,
        payload=payload,
    )
    if facet_label is None or value_text is None:
        return None
    subject_surface = _subject_surface(memory_row, payload)
    if subject_surface is None:
        return None

    observed_at = _optional_text(source_span.get("occurred_at")) or _optional_text(
        memory_row.get("created_at")
    )
    valid_from = _optional_text(memory_row.get("valid_from"))
    valid_to = _optional_text(memory_row.get("valid_to"))
    temporal_confidence = _temporal_confidence(payload, item)
    temporal_type = _optional_text(memory_row.get("temporal_type"))
    temporal_is_known = bool(
        valid_from
        or valid_to
        or temporal_confidence is not None
        or (temporal_type is not None and temporal_type != "unknown")
    )

    return SourceBackedFactFacetProjection(
        memory_id=memory_id,
        source_span_id=source_span_id,
        source_message_id=source_message_id,
        conversation_id=_optional_text(source_span.get("conversation_id"))
        or _optional_text(memory_row.get("conversation_id")),
        subject_surface=subject_surface,
        subject_cluster_id=_optional_text(memory_row.get("presence_cluster_id")),
        surface_class=surface_class,
        facet_label=facet_label,
        value_text=value_text,
        value_type=_structured_text(payload, ("value_type",)) or "text",
        assertion_kind=_structured_text(payload, ("assertion_kind",))
        or object_type.value,
        list_group_key=_structured_text(payload, ("list_group_key",)) or facet_label,
        support_kind=_optional_text(evidence_packet.get("support_kind") if evidence_packet else None)
        or "direct",
        observed_at=observed_at,
        valid_from=valid_from,
        valid_to=valid_to,
        current_state=valid_to is None,
        supersedes_fact_id=_structured_text(payload, ("supersedes_fact_id",)),
        temporal_phrase=_structured_text(payload, ("temporal_phrase",)),
        temporal_anchor_at=_structured_text(payload, ("temporal_anchor_at",))
        or (observed_at if temporal_is_known else None),
        resolved_interval_start=valid_from,
        resolved_interval_end=valid_to,
        temporal_granularity=_structured_text(payload, ("temporal_granularity",)),
        temporal_resolution_type=_structured_text(payload, ("temporal_resolution_type",))
        or (temporal_type if temporal_is_known else None),
        temporal_confidence=temporal_confidence,
        language_code=_first_language_code(memory_row, item),
        confidence=float(item.confidence),
    )


def _first_source_span(evidence_packet: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(evidence_packet, dict):
        return None
    spans = evidence_packet.get("spans")
    if not isinstance(spans, list):
        return None
    for span in spans:
        if not isinstance(span, dict):
            continue
        if span.get("span_role") == "source":
            return span
    return None


def _facet_value_for_item(
    *,
    item: ExtractedMemoryBase,
    object_type: MemoryObjectType,
    payload: dict[str, Any],
) -> tuple[str | None, str | None, str]:
    if isinstance(item, ExtractedBelief):
        return item.claim_key, item.claim_value, "structured"
    structured_facet_label = _structured_text(
        payload,
        ("facet_label",),
    )
    structured_value_text = _structured_text(
        payload,
        ("value_text",),
    )
    if structured_facet_label is None or structured_value_text is None:
        return None, None, "generic"
    return structured_facet_label, structured_value_text, "structured"


def _subject_surface(
    memory_row: dict[str, Any],
    payload: dict[str, Any],
) -> str | None:
    direct = _structured_text(
        payload,
        (
            "subject_surface",
            "subject",
        ),
    )
    if direct is not None:
        return direct
    presence = payload.get("presence_attribution")
    if isinstance(presence, dict):
        source = presence.get("source")
        if isinstance(source, dict):
            display = _optional_text(source.get("display_name"))
            if display is not None:
                return display
        active = presence.get("active")
        if isinstance(active, dict):
            display = _optional_text(active.get("display_name"))
            if display is not None:
                return display
    return None


def _structured_text(
    payload: dict[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        value = payload.get(key)
        text = _optional_text(value)
        if text is not None:
            return text
    return None


def _first_language_code(
    memory_row: dict[str, Any],
    item: ExtractedMemoryBase,
) -> str | None:
    row_codes = memory_row.get("language_codes_json")
    if isinstance(row_codes, list) and row_codes:
        return _optional_text(row_codes[0])
    if item.language_codes:
        return _optional_text(item.language_codes[0])
    return None


def _temporal_confidence(
    payload: dict[str, Any],
    item: ExtractedMemoryBase,
) -> float | None:
    value = payload.get("temporal_confidence")
    try:
        resolved = float(value if value is not None else item.temporal_confidence)
    except (TypeError, ValueError):
        return None
    if resolved <= 0.0:
        return None
    return max(0.0, min(1.0, resolved))


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


__all__ = [
    "FACT_FACET_PROJECTION_SCHEMA_VERSION",
    "SourceBackedFactFacetProjection",
    "lean_result_to_extraction_result",
    "source_backed_fact_facet_projection",
]
