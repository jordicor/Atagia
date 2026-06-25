"""Shared test helper: convert rich extraction fixtures into lean wire JSON.

The extractor now requests :class:`LeanExtractionResult` from the model, so the
canned providers used throughout the test suite must emit the lean shape instead
of the historical rich ``ExtractionResult`` shape. Fixtures stay authored in the
readable rich shape; these helpers translate them at the provider boundary.

Fields the lean contract no longer carries are dropped here and are supplied by
the server-side mapper (``atagia.memory.extraction_mapping``) at validation time.
"""

from __future__ import annotations

import json
from typing import Any

MEMORY_EXTRACTION_CARD_PURPOSES = frozenset(
    {
        "memory_extraction_candidate_card",
        "memory_extraction_kind_scope_card",
        "memory_extraction_evidence_card",
        "memory_extraction_index_card",
        "memory_extraction_temporal_card",
        "memory_extraction_belief_card",
        "memory_extraction_coverage_members_card",
    }
)

_BUCKET_TO_LEAN_KIND = {
    "evidences": "evidence",
    "beliefs": "belief",
    "contract_signals": "contract_signal",
    "state_updates": "state_update",
}
# Mirrors MemoryExtractor._canonical_write_scope so a converted fixture persists
# under the same canonical scope the extractor would resolve from a legacy value.
_SCOPE_TO_SUBJECT_SCOPE = {
    "conversation": "chat",
    "ephemeral_session": "chat",
    "chat": "chat",
    "workspace": "character",
    "character": "character",
    "global_user": "user",
    "assistant_mode": "user",
    "user": "user",
}
_LEAN_PASSTHROUGH_FIELDS = (
    "canonical_text",
    "confidence",
    "language_codes",
    "index_text",
    "preserve_verbatim",
    "support_kind",
    "claim_key",
    "claim_value",
    "coverage_members",
)


def _rich_item_to_lean_candidate(
    item: dict[str, Any],
    *,
    kind: str,
    default_language_codes: bool,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {"kind": kind}
    for field in _LEAN_PASSTHROUGH_FIELDS:
        if field in item:
            candidate[field] = item[field]
    # When language_codes are absent the rich model defaulted them to [] and let
    # the after-validator raise; preserve that exact failure path unless the
    # caller asks for the convenience English default.
    candidate.setdefault("language_codes", ["en"] if default_language_codes else [])
    scope_value = str(item.get("scope") or "chat")
    candidate["subject_scope"] = _SCOPE_TO_SUBJECT_SCOPE.get(scope_value, "chat")
    if "source_quote" in item:
        candidate["source_span"] = item["source_quote"]
    temporal_type = item.get("temporal_type")
    valid_from = item.get("valid_from_iso")
    valid_to = item.get("valid_to_iso")
    if temporal_type is not None or valid_from is not None or valid_to is not None:
        temporal_status: dict[str, Any] = {}
        if temporal_type is not None:
            temporal_status["type"] = temporal_type
        if valid_from is not None:
            temporal_status["valid_from_iso"] = valid_from
        if valid_to is not None:
            temporal_status["valid_to_iso"] = valid_to
        candidate["temporal_status"] = temporal_status
    return candidate


def rich_extraction_payload_to_lean(
    payload: dict[str, Any],
    *,
    default_language_codes: bool = True,
) -> dict[str, Any]:
    """Translate a rich ``ExtractionResult``-shaped dict into a lean wire dict.

    With ``default_language_codes=False`` a missing ``language_codes`` field is
    carried as an empty list rather than defaulted to English, so the lean
    validators raise exactly as the rich validators did (used by retry tests).
    """

    candidates: list[dict[str, Any]] = []
    for bucket, kind in _BUCKET_TO_LEAN_KIND.items():
        items = payload.get(bucket)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            candidates.append(
                _rich_item_to_lean_candidate(
                    item,
                    kind=kind,
                    default_language_codes=default_language_codes,
                )
            )
    lean: dict[str, Any] = {"candidates": candidates}
    if "nothing_durable" in payload:
        lean["nothing_durable"] = bool(payload["nothing_durable"]) and not candidates
    return lean


def is_rich_extraction_payload(payload: Any) -> bool:
    """Return True when the dict looks like a rich extraction result, not lean."""

    if not isinstance(payload, dict):
        return False
    if "candidates" in payload:
        return False
    return any(bucket in payload for bucket in _BUCKET_TO_LEAN_KIND)


def rich_extraction_json_to_lean(output_text: str) -> str:
    """Convert a JSON string carrying a rich extraction payload into lean JSON.

    Non-JSON or non-extraction payloads are returned unchanged, so this is safe
    to apply to a provider's combined output stream.
    """

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError:
        return output_text
    if not is_rich_extraction_payload(payload):
        return output_text
    return json.dumps(rich_extraction_payload_to_lean(payload))


def is_memory_extraction_card_purpose(purpose: object) -> bool:
    return str(purpose) in MEMORY_EXTRACTION_CARD_PURPOSES


def memory_extraction_card_output_from_payload(
    payload: dict[str, Any] | str,
    purpose: object,
) -> str:
    """Render a rich or lean extraction fixture as one plain-text card output."""

    if isinstance(payload, str):
        try:
            parsed_payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload
    else:
        parsed_payload = payload
    if is_rich_extraction_payload(parsed_payload):
        lean_payload = rich_extraction_payload_to_lean(parsed_payload)
    elif isinstance(parsed_payload, dict):
        lean_payload = parsed_payload
    else:
        return "none"

    candidates = lean_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return "none"

    purpose_text = str(purpose)
    if purpose_text == "memory_extraction_candidate_card":
        lines = []
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            canonical_text = str(candidate.get("canonical_text") or "").strip()
            if canonical_text:
                lines.append(f"cand_{index:03d} | {canonical_text}")
        return "\n".join(lines) or "none"

    lines = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        candidate_id = f"cand_{index:03d}"
        if purpose_text == "memory_extraction_kind_scope_card":
            kind = str(candidate.get("kind") or "evidence")
            scope = str(candidate.get("subject_scope") or "user")
            confidence = candidate.get("confidence", 0.75)
            lines.append(f"{candidate_id} {kind} {scope} {confidence}")
        elif purpose_text == "memory_extraction_evidence_card":
            support_kind = str(candidate.get("support_kind") or "direct")
            preserve_verbatim = (
                "true" if bool(candidate.get("preserve_verbatim")) else "false"
            )
            raw_languages = candidate.get("language_codes") or ("en",)
            if isinstance(raw_languages, str):
                language_codes = raw_languages
            else:
                language_codes = ",".join(str(item) for item in raw_languages) or "en"
            source_span = str(
                candidate.get("source_span")
                or candidate.get("canonical_text")
                or "none"
            )
            lines.append(
                f"{candidate_id} {support_kind} {preserve_verbatim} "
                f"{language_codes} | {source_span}"
            )
        elif purpose_text == "memory_extraction_index_card":
            index_text = str(candidate.get("index_text") or "none")
            lines.append(f"{candidate_id} | {index_text}")
        elif purpose_text == "memory_extraction_temporal_card":
            temporal_status = candidate.get("temporal_status")
            if isinstance(temporal_status, dict):
                temporal_type = str(temporal_status.get("type") or "none")
                valid_from = str(temporal_status.get("valid_from_iso") or "none")
                valid_to = str(temporal_status.get("valid_to_iso") or "none")
                lines.append(f"{candidate_id} {temporal_type} {valid_from} {valid_to}")
            else:
                lines.append(f"{candidate_id} none none")
        elif purpose_text == "memory_extraction_belief_card":
            if str(candidate.get("kind") or "") == "belief":
                claim_key = str(candidate.get("claim_key") or "none")
                claim_value = str(candidate.get("claim_value") or "none")
                lines.append(f"{candidate_id} {claim_key} {claim_value}")
            else:
                lines.append(f"{candidate_id} none none")
        elif purpose_text == "memory_extraction_coverage_members_card":
            members = candidate.get("coverage_members")
            members_json = json.dumps(members) if isinstance(members, list) else "[]"
            lines.append(f"{candidate_id} | {members_json}")
    return "\n".join(lines) or "none"
