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
import html
import re
from typing import Any
from datetime import datetime

from atagia.core.source_references import SourceReferenceCatalog

MEMORY_EXTRACTION_CARD_PURPOSES = frozenset(
    {
        "memory_extraction_candidate_card",
        "memory_extraction_kind_card",
        "memory_extraction_scope_card",
        "memory_extraction_confidence_card",
        "memory_extraction_evidence_support_card",
        "memory_extraction_preserve_verbatim_card",
        "memory_extraction_candidate_language_card",
        "memory_extraction_source_reference_card",
        "memory_extraction_index_card",
        "memory_extraction_belief_key_card",
        "memory_extraction_belief_value_card",
        "memory_extraction_temporal_type_card",
        "memory_extraction_temporal_interval_card",
        "memory_date_resolution",
        "memory_extraction_coverage_members_card",
        "memory_extraction_coverage_member_identity_card",
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
    "date_resolution_output",
    "date_resolution_outputs",
    "temporal_interval_output",
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
    *,
    prompt: str,
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

    if purpose_text == "memory_date_resolution":
        text = prompt.split("\nText: ", 1)[1].split("\n\n", 1)[0]
        endpoint_answers = [
            candidate["date_resolution_outputs"][text]
            for candidate in candidates
            if text in candidate.get("date_resolution_outputs", {})
        ]
        if endpoint_answers:
            if len(endpoint_answers) != 1:
                raise ValueError("Date fixture must identify one endpoint response")
            return str(endpoint_answers[0])
        matches = [candidate for candidate in candidates if candidate.get("canonical_text") == text]
        if len(matches) != 1:
            raise ValueError("Date fixture must identify its exact candidate")
        candidate = matches[0]
        if "date_resolution_output" in candidate:
            return str(candidate["date_resolution_output"])
        temporal_status = candidate.get("temporal_status") or {}
        timestamp = temporal_status.get("valid_from_iso") or temporal_status.get("valid_to_iso")
        return f"exact|{datetime.fromisoformat(timestamp).date().isoformat()}|days|0" if timestamp else "unknown"

    if purpose_text in {
        "memory_extraction_belief_key_card",
        "memory_extraction_belief_value_card",
    }:
        candidate = _belief_candidate_shown_in_prompt(candidates, prompt)
        field = (
            "claim_key"
            if purpose_text == "memory_extraction_belief_key_card"
            else "claim_value"
        )
        value = candidate.get(field)
        if candidate.get("kind") != "belief" or not isinstance(value, str) or not value.strip():
            raise ValueError(f"Belief fixture requires {field} for the shown candidate")
        return value

    if purpose_text in {
        "memory_extraction_coverage_members_card",
        "memory_extraction_coverage_member_identity_card",
    }:
        return coverage_card_output_from_lean_candidates(candidates, purpose_text, prompt)

    if purpose_text in {
        "memory_extraction_kind_card",
        "memory_extraction_scope_card",
        "memory_extraction_confidence_card",
    }:
        candidate = _classification_candidate_shown_in_prompt(candidates, prompt)
        if purpose_text == "memory_extraction_kind_card":
            return str(candidate.get("kind") or "evidence")
        if purpose_text == "memory_extraction_scope_card":
            return str(candidate.get("subject_scope") or "user")
        return str(candidate.get("confidence", 0.75))

    if purpose_text in {
        "memory_extraction_evidence_support_card",
        "memory_extraction_preserve_verbatim_card",
        "memory_extraction_candidate_language_card",
        "memory_extraction_source_reference_card",
    }:
        return _evidence_answer_from_payload(candidates, purpose_text, prompt)

    candidates = _candidates_shown_in_prompt(candidates, prompt)

    lines = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("_fixture_candidate_id") or f"cand_{index:03d}")
        if purpose_text == "memory_extraction_index_card":
            index_text = str(candidate.get("index_text") or "none")
            lines.append(f"{candidate_id} | {index_text}")
        elif purpose_text == "memory_extraction_temporal_type_card":
            temporal_status = candidate.get("temporal_status")
            if isinstance(temporal_status, dict):
                temporal_type = str(temporal_status.get("type") or "none")
                lines.append(temporal_type)
            else:
                lines.append("none")
        elif purpose_text == "memory_extraction_temporal_interval_card":
            temporal_status = candidate.get("temporal_status")
            if not isinstance(temporal_status, dict):
                raise ValueError("Interval requested without temporal fixture status")
            if "temporal_interval_output" in candidate:
                lines.extend(str(candidate["temporal_interval_output"]).splitlines())
                continue
            timestamps = [
                datetime.fromisoformat(value) if value else None
                for value in (temporal_status.get("valid_from_iso"), temporal_status.get("valid_to_iso"))
            ]
            start, end = timestamps
            # Mechanical conversion of already-specified fixture bounds, never
            # interpretation of source wording or production date resolution.
            period = "day"
            if start and end and start.date() != end.date():
                if start.weekday() == 0 and end.weekday() == 6 and (end.date() - start.date()).days == 6:
                    period = "week"
                else:
                    raise ValueError("Multi-day fixture must supply explicit temporal_interval_output")
            for timestamp in timestamps:
                if timestamp is None:
                    lines.append("null")
                    continue
                stamp = timestamp.isoformat()
                offset = stamp[-6:] if timestamp.utcoffset() is not None else None
                lines.append(json.dumps({
                    "text": None, "time": timestamp.time().isoformat(),
                    "offset": offset, "period": period,
                }))
    return "\n".join(lines) or "none"


def coverage_card_output_from_lean_candidates(
    candidates: list[Any], purpose: str, prompt: str
) -> str:
    """Answer a single known candidate's coverage fixture calls."""

    match = re.search(r"<candidate>\n(.*?)\n</candidate>", prompt, re.DOTALL)
    if match is None:
        raise ValueError("Coverage request has no candidate")
    candidate_text = html.unescape(match.group(1))
    matches = [
        candidate for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("canonical_text") == candidate_text
    ]
    if len(matches) != 1:
        raise ValueError("Coverage fixture cannot identify its candidate")
    members = matches[0].get("coverage_members") or []
    if purpose == "memory_extraction_coverage_members_card":
        return "\n".join(
            json.dumps(member["display_text"], ensure_ascii=False)
            for member in members
        ) or "none"
    member_match = re.search(r"<member>\n(.*?)\n</member>", prompt, re.DOTALL)
    if member_match is None:
        raise ValueError("Identity request has no member")
    member_name = html.unescape(member_match.group(1))
    identities = [
        member["member_key"] for member in members
        if member["display_text"] == member_name
    ]
    if len(identities) != 1:
        raise ValueError("Coverage fixture cannot identify its member")
    return str(identities[0])


def _evidence_answer_from_payload(
    candidates: list[Any], purpose: str, prompt: str
) -> str:
    match = re.search(r"<candidate>\n(.*?)\n</candidate>", prompt, re.DOTALL)
    if match is None:
        raise ValueError("Evidence request has no known candidate")
    candidate_text = html.unescape(match.group(1))
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("canonical_text") == candidate_text
    ]
    if not matches:
        raise ValueError("Fixture has no candidate shown in the request")
    candidate = matches[0]
    if purpose == "memory_extraction_evidence_support_card":
        if "source_span" in candidate and candidate["source_span"] is None:
            return "none"
        return str(candidate.get("support_kind") or "direct")
    if purpose == "memory_extraction_preserve_verbatim_card":
        return "yes" if candidate.get("preserve_verbatim") else "no"
    if purpose == "memory_extraction_candidate_language_card":
        languages = candidate.get("language_codes") or ["en"]
        return (
            "\n".join(languages)
            if isinstance(languages, list | tuple)
            else str(languages)
        )
    if purpose != "memory_extraction_source_reference_card":
        raise ValueError("Unknown evidence purpose")
    catalog = _source_catalog_from_prompt(prompt)
    source_span = candidate.get("source_span")
    if "source_span" in candidate and source_span is None:
        return "none"
    if source_span is None:
        selected = catalog.anchors
    else:
        quote = str(source_span)
        quote_start = catalog.source_text.find(quote)
        if not quote or quote_start < 0:
            raise ValueError(f"Fixture quote is absent from source: {quote!r}")
        quote_end = quote_start + len(quote)
        selected = tuple(
            anchor for anchor in catalog.anchors
            if anchor.char_end > quote_start and anchor.char_start < quote_end
        )
    if not selected:
        raise ValueError("Fixture source has no referenceable text")
    return f"{selected[0].reference_id} {selected[-1].reference_id}"


def _source_catalog_from_prompt(prompt: str) -> SourceReferenceCatalog:
    """Read the displayed source units from an evidence request exactly."""

    match = re.search(r"<message_text>\n(.*?)\n</message_text>", prompt, re.DOTALL)
    if match is None:
        raise ValueError("Evidence request does not contain a source message")
    rendered = match.group(1)
    source_text = html.unescape(re.sub(r"\[r[1-9][0-9]*\]", "", rendered))
    catalog = SourceReferenceCatalog(source_text)
    if catalog.render() != rendered:
        raise ValueError("Evidence request source markers are inconsistent")
    return catalog


def _candidates_shown_in_prompt(
    candidates: list[Any], prompt: str
) -> list[dict[str, Any]]:
    """Honor candidate deduplication and bounded-output caps in the request."""

    match = re.search(r"<candidates>\n(.*?)\n</candidates>", prompt, re.DOTALL)
    if match is None:
        raise ValueError("Enrichment request has no candidate block")
    by_text: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if isinstance(candidate, dict):
            by_text.setdefault(str(candidate.get("canonical_text")), candidate)
    selected: list[dict[str, Any]] = []
    for line in match.group(1).splitlines():
        row = re.fullmatch(r"cand_[0-9]{3}: (.*)", line)
        if row is None:
            raise ValueError("Enrichment request has a malformed candidate row")
        candidate_text = html.unescape(row.group(1))
        if candidate_text not in by_text:
            raise ValueError("Fixture has no candidate shown in the request")
        selected.append({**by_text[candidate_text], "_fixture_candidate_id": line.split(":", 1)[0]})
    return selected


def _belief_candidate_shown_in_prompt(
    candidates: list[Any], prompt: str
) -> dict[str, Any]:
    match = re.search(r"<candidate>\n(.*?)\n</candidate>", prompt, re.DOTALL)
    if match is None:
        raise ValueError("Belief request has no candidate")
    candidate_text = html.unescape(match.group(1))
    matches = [
        candidate for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("canonical_text") == candidate_text
    ]
    if len(matches) != 1:
        raise ValueError("Belief request must show exactly one fixture candidate")
    return matches[0]

def _classification_candidate_shown_in_prompt(
    candidates: list[Any], prompt: str
) -> dict[str, Any]:
    match = re.search(r"<candidate_text>\n(.*?)\n</candidate_text>", prompt, re.DOTALL)
    if match is None:
        raise ValueError("Classification request has no candidate text")
    candidate_text = html.unescape(match.group(1))
    matches = [
        candidate for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("canonical_text") == candidate_text
    ]
    if len(matches) != 1:
        raise ValueError("Fixture must identify exactly one classification candidate")
    return matches[0]
