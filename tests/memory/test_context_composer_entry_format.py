"""Tests for the trimmed visible per-memory entry format.

Each admitted memory renders as a numbered entry whose parenthetical header
carries a kind tag and the resolved point date or validity interval. Internal
bookkeeping (confidence, scope, privacy fields, raw temporal windows) stays
in the data model but must never reach the prompt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from atagia.core.clock import FrozenClock
from atagia.memory.context_composer import ContextComposer
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import ScoredCandidate

MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


def _resolved_policy(context_budget_tokens: int = 600):
    loader = ManifestLoader(MANIFESTS_DIR)
    manifest = loader.load_all()["coding_debug"]
    resolved = PolicyResolver().resolve(manifest, None, None)
    return resolved.model_copy(update={"context_budget_tokens": context_budget_tokens})


def _candidate(
    memory_id: str,
    *,
    final_score: float,
    canonical_text: str,
    object_type: str = "evidence",
    resolved_date: str | None = None,
    extra_fields: dict | None = None,
) -> ScoredCandidate:
    memory_object = {
        "id": memory_id,
        "object_type": object_type,
        "canonical_text": canonical_text,
        "payload_json": {},
    }
    if extra_fields:
        memory_object.update(extra_fields)
    return ScoredCandidate(
        memory_id=memory_id,
        memory_object=memory_object,
        llm_applicability=0.7,
        retrieval_score=0.6,
        vitality_boost=0.2,
        confirmation_boost=0.0,
        need_boost=0.0,
        penalty=0.0,
        final_score=final_score,
        resolved_date=resolved_date,
    )


def _composer() -> ContextComposer:
    return ContextComposer(
        FrozenClock(datetime(2026, 3, 30, 22, 0, tzinfo=timezone.utc))
    )


def _compose(candidates: list[ScoredCandidate], **options):
    return _composer().compose(
        scored_candidates=candidates,
        current_contract={},
        user_state=None,
        resolved_policy=_resolved_policy(),
        conversation_messages=[],
        **options,
    )


def _entry_header(memory_block: str, text_marker: str) -> str:
    lines = memory_block.splitlines()
    text_line = next(line for line in lines if text_marker in line)
    return lines[lines.index(text_line) - 1]


def test_full_metadata_entry_renders_only_kind_and_date() -> None:
    context = _compose(
        [
            _candidate(
                "mem_full",
                final_score=0.9,
                canonical_text="Caroline attended the conference on a Saturday.",
                resolved_date="2025-03-08",
                extra_fields={
                    "confidence": 0.86,
                    "scope": "user",
                    "privacy_level": 1,
                    "memory_category": "contact_identity",
                    "preserve_verbatim": True,
                    "intimacy_boundary": "ordinary",
                    "temporal_type": "event_triggered",
                    "valid_from": "2025-03-07T00:00:00+00:00",
                    "valid_to": "2025-03-09T00:00:00+00:00",
                    "payload_json": {
                        "source_message_window_start_occurred_at": "2025-03-09T10:00:00+00:00",
                        "source_message_window_end_occurred_at": "2025-03-09T10:05:00+00:00",
                    },
                },
            )
        ]
    )

    header = _entry_header(context.memory_block, "Caroline attended")
    assert header == "1. (evidence, date: from 2025-03-07T00:00:00+00:00 through 2025-03-09T00:00:00+00:00)"
    for dropped in (
        "confidence",
        "scope",
        "privacy_level",
        "memory_category",
        "preserve_verbatim",
        "intimacy_boundary",
        "event_time",
        "valid_window",
        "source_window",
        "resolved_date",
    ):
        assert dropped not in context.memory_block
    assert "?" not in header


def test_entry_preserves_both_legacy_validity_bounds() -> None:
    context = _compose(
        [
            _candidate(
                "mem_bounded",
                final_score=0.9,
                canonical_text="User painted a lake sunrise.",
                extra_fields={
                    "temporal_type": "bounded",
                    "valid_from": "2041-05-15T00:00:00+00:00",
                    "valid_to": "2041-05-31T23:59:59+00:00",
                },
            )
        ]
    )

    header = _entry_header(context.memory_block, "User painted")
    assert header == "1. (evidence, date: from 2041-05-15T00:00:00+00:00 through 2041-05-31T23:59:59+00:00)"


def test_entry_labels_source_timestamp_without_resolving_an_event_date() -> None:
    context = _compose(
        [
            _candidate(
                "mem_window",
                final_score=0.9,
                canonical_text="user: My calibration targets are cobalt and quartz",
                extra_fields={
                    "payload_json": {
                        "source_kind_variant": "conversation_window",
                        "source_message_window_start_occurred_at": "2026-04-04T11:00:00+00:00",
                        "source_message_window_end_occurred_at": "2026-04-04T11:00:00+00:00",
                    },
                },
            )
        ]
    )

    header = _entry_header(context.memory_block, "calibration targets")
    assert header == "1. (evidence, date: source timestamp: 2026-04-04T11:00:00+00:00)"


def test_entry_without_any_date_renders_no_date_or_placeholder() -> None:
    context = _compose(
        [
            _candidate(
                "mem_timeless",
                final_score=0.9,
                canonical_text="User prefers direct answers.",
            )
        ]
    )

    header = _entry_header(context.memory_block, "User prefers")
    assert header == "1. (evidence)"
    assert "date:" not in header
    assert "?" not in header


def test_entry_source_quote_rendering_is_preserved() -> None:
    context = _compose(
        [
            _candidate(
                "mem_quoted",
                final_score=0.9,
                canonical_text="Nia's observatory access badge expired.",
                extra_fields={
                    "evidence_packets": [
                        {
                            "support_kind": "direct",
                            "spans": [
                                {
                                    "span_role": "source",
                                    "quote_text": (
                                        "My observatory access badge expired yesterday."
                                    ),
                                }
                            ],
                        }
                    ],
                },
            )
        ]
    )

    assert (
        "source_quote: My observatory access badge expired yesterday."
        in context.memory_block
    )


def test_entry_format_is_uniform_across_object_types() -> None:
    context = _compose(
        [
            _candidate(
                "mem_evidence",
                final_score=0.9,
                canonical_text="Literal user utterance.",
                object_type="evidence",
                resolved_date="2025-03-08",
            ),
            _candidate(
                "mem_belief",
                final_score=0.8,
                canonical_text="Derived stable preference.",
                object_type="belief",
                resolved_date="2025-03-08",
            ),
            _candidate(
                "mem_summary",
                final_score=0.7,
                canonical_text="Episode summary view.",
                object_type="summary_view",
                resolved_date="2025-03-08",
            ),
        ]
    )

    assert _entry_header(context.memory_block, "Literal user utterance.") == (
        "1. (evidence, date: 2025-03-08)"
    )
    assert _entry_header(context.memory_block, "Derived stable preference.") == (
        "2. (belief, date: 2025-03-08)"
    )
    assert _entry_header(context.memory_block, "Episode summary view.") == (
        "3. (summary_view, date: 2025-03-08)"
    )


def test_uncertain_persisted_date_stays_approximate_in_memory_and_answer_evidence() -> None:
    candidate = _candidate(
        "mem_quoted", final_score=0.9,
        canonical_text="Nia's observatory access badge expired.",
        resolved_date="2025-03-08",
        extra_fields={
            "evidence_packets": [{
                "support_kind": "direct",
                "spans": [{
                    "span_role": "source",
                    "quote_text": "My observatory access badge expired yesterday.",
                }],
            }],
        },
    ).model_copy(update={"date_resolution_status": "completed", "date_certainty": "uncertain"})
    context = _compose([candidate])
    assert "date: approximately 2025-03-08 (representative date)" in context.memory_block
    context = _compose([candidate], query_type="temporal", enable_final_answer_evidence_pack=True)
    assert "date: approximately 2025-03-08 (representative date)" in context.answer_evidence_block
    assert "date_certainty: uncertain" in context.answer_evidence_block
    assert context.answer_evidence_items[0]["normalization"]["date_certainty"] == "uncertain"


def test_pending_date_is_distinct_from_completed_unknown_in_answer_evidence() -> None:
    for status, certainty, label in (
        ("completed", "unknown", "unknown"),
        ("pending_analysis", None, "pending analysis"),
        ("stale", None, "stale"),
        ("unprocessed", None, "source timestamp: 2025-03-09T10:00:00+00:00"),
    ):
        candidate = _candidate(
            "mem_quoted", final_score=0.9,
            canonical_text="Nia's observatory access badge expired.",
            extra_fields={
                "occurred_at": "2025-03-09T10:00:00+00:00",
                "evidence_packets": [{
                    "support_kind": "direct",
                    "spans": [{
                        "span_role": "source",
                        "quote_text": "My observatory access badge expired yesterday.",
                        "occurred_at": "2025-03-09T10:00:00+00:00",
                    }],
                }],
            },
        ).model_copy(update={"date_resolution_status": status, "date_certainty": certainty})
        context = _compose([candidate])
        header = _entry_header(context.memory_block, "Nia's observatory")
        expected_date = label if status == "unprocessed" else ""
        assert header == (
            f"1. (evidence, date: {expected_date})" if expected_date else "1. (evidence)"
        )
        context = _compose([candidate], query_type="temporal", enable_final_answer_evidence_pack=True)
        assert f"- date: {label}" in context.answer_evidence_block
        assert f"- date_resolution_status: {status}" in context.answer_evidence_block
        assert context.answer_evidence_items[0]["date"] == expected_date
