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
