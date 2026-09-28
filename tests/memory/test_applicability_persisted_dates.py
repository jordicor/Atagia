"""Persisted date annotations are reused independently of query relevance."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings, default_resource_path
from atagia.core.db_sqlite import close_connection, initialize_database
from atagia.core.repositories import MemoryObjectRepository, UserRepository
from atagia.memory.applicability_scorer import ApplicabilityScorer
from atagia.memory.context_composer import ContextComposer
from atagia.memory.date_resolution import DateResolution, parse_date_resolution
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = ROOT / "src" / "atagia" / "resources" / "migrations"
MANIFESTS_DIR = ROOT / "src" / "atagia" / "resources" / "manifests"


class DateProvider(LLMProvider):
    name = "date-cache-test"

    def __init__(self) -> None:
        self.purposes: list[str] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata["purpose"])
        self.purposes.append(purpose)
        assert purpose == "applicability_relevance_card"
        output = "candidate_000 exact"
        return LLMCompletionResponse(
            provider=self.name, model=request.model, output_text=output
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Persisted date tests do not use embeddings")


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=default_resource_path("migrations"),
        manifests_path=default_resource_path("manifests"),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )


def _context(user_id: str) -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id=user_id,
        conversation_id=f"conversation_{user_id}",
        source_message_id=f"query_{user_id}",
        workspace_id=None,
        assistant_mode_id="coding_debug",
        recent_messages=[],
    )


def _candidate(row: dict[str, object]) -> dict[str, object]:
    return {**row, "rrf_score": 0.7, "rank": 1, "retrieval_sources": ["fts"]}


_TEXT = "Two days ago I signed the lease."
_SOURCE_DATE = "2025-03-10T09:00:00+00:00"


def _annotation(answer: str) -> dict[str, object]:
    parsed = parse_date_resolution(answer, _SOURCE_DATE)
    return DateResolution(
        **parsed.model_dump(),
        source_text_sha256=sha256(_TEXT.encode("utf-8")).hexdigest(),
        reference_date="2025-03-10",
    ).model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "certainty", "resolved", "status"),
    [
        ("exact|ref|days|-2", "exact", "2025-03-08", "completed"),
        ("uncertain|ref|days|-2", "uncertain", "2025-03-08", "completed"),
        ("unknown", "unknown", None, "completed"),
        ("analyze", None, None, "pending_analysis"),
    ],
)
async def test_persisted_date_reloads_without_date_calls(
    answer: str, certainty: str | None, resolved: str | None, status: str,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        clock = FrozenClock(datetime(2026, 3, 30, 21, 0, tzinfo=timezone.utc))
        await UserRepository(connection, clock).create_user("usr_1")
        memories = MemoryObjectRepository(connection, clock)
        await memories.create_memory_object(
            user_id="usr_1", object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER, canonical_text=_TEXT,
            source_kind=MemorySourceKind.EXTRACTED, confidence=0.9,
            privacy_level=0, memory_id="memory_1", temporal_type="event_triggered",
            valid_from=None, valid_to=None,
            payload={
                "source_message_ids": ["source_1"],
                "source_occurred_at": _SOURCE_DATE,
                "source_message_window_start_occurred_at": _SOURCE_DATE,
                "source_message_window_end_occurred_at": _SOURCE_DATE,
                "date_resolution": _annotation(answer),
            },
        )
        provider = DateProvider()
        policy = PolicyResolver().resolve(
            ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"], None, None,
        )
        for query in ("When did I sign the lease?", "What happened around that time?"):
            row = await memories.get_memory_object("memory_1", "usr_1")
            assert row is not None
            # A fresh client/scorer proves reuse comes from SQLite, not a process cache.
            scorer = ApplicabilityScorer(
                LLMClient(provider_name=provider.name, providers=[provider]), clock, _settings(),
            )
            result = await scorer.score_shortlist(
                [_candidate(row)], message_text=query,
                conversation_context=_context("usr_1"),
                resolved_policy=policy, detected_needs=[],
            )
            candidate = result[0]
            assert candidate.llm_applicability == 0.95
            assert candidate.resolved_date == resolved
            assert candidate.date_certainty == certainty
            assert candidate.date_resolution_status == status
            assert candidate.memory_object["valid_to"] is None
            assert candidate.penalty == 0
            rendered = ContextComposer._answer_evidence_date(candidate, row)
            if certainty == "uncertain":
                assert rendered == "approximately 2025-03-08 (representative date)"
            elif resolved:
                assert rendered == resolved
            else:
                # Unknown/pending event timing must not become the source timestamp.
                assert rendered == ""
            normalization = ContextComposer._answer_evidence_normalization(
                candidate, memory_object=row, source_messages_by_id={}, quote_source="source_message",
            )
            assert normalization["date_resolution_status"] == status
            assert normalization.get("date_certainty") == certainty
        assert provider.purposes == ["applicability_relevance_card"] * 2
    finally:
        await close_connection(connection)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    "missing", "text", "anchor", "malformed", "not_applicable",
    "not_applicable_text", "not_applicable_anchor", "not_applicable_legacy",
])
async def test_unprocessed_or_stale_dates_never_run_a_retrieval_resolver(change: str) -> None:
    provider = DateProvider()
    clock = FrozenClock(datetime(2026, 3, 30, tzinfo=timezone.utc))
    scorer = ApplicabilityScorer(
        LLMClient(provider_name=provider.name, providers=[provider]), clock, _settings(),
    )
    policy = PolicyResolver().resolve(
        ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"], None, None,
    )
    payload = {"source_occurred_at": _SOURCE_DATE, "date_resolution": _annotation("exact|ref|days|-2")}
    text = _TEXT
    status = "stale"
    if change == "missing":
        payload.pop("date_resolution")
        status = "unprocessed"
    elif change.startswith("not_applicable"):
        payload.pop("date_resolution")
        payload["date_resolution_not_applicable"] = {
            "source_text_sha256": sha256(_TEXT.encode("utf-8")).hexdigest(),
            "reference_date": "2025-03-10",
        }
        if change == "not_applicable":
            status = "not_applicable"
        elif change == "not_applicable_text":
            text = "Yesterday I signed the lease."
        elif change == "not_applicable_anchor":
            payload["source_occurred_at"] = "2025-03-11T09:00:00+00:00"
        elif change == "not_applicable_legacy":
            payload["date_resolution_not_applicable"] = True
    elif change == "text":
        text = "Yesterday I signed the lease."
    elif change == "anchor":
        payload["source_occurred_at"] = "2025-03-11T09:00:00+00:00"
    elif change == "malformed":
        payload["date_resolution"] = {"status": "completed"}
    row = {
        "id": "memory_1", "user_id": "usr_1", "canonical_text": text,
        "payload_json": payload, "object_type": "evidence", "scope": "user",
        "scope_canonical": "user", "status": "active", "privacy_level": 0,
        "rrf_score": 0.7, "retrieval_sources": ["fts"],
    }
    for query in ("When did I sign?", "What date?"):
        result = await scorer.score_shortlist(
            [row], message_text=query, conversation_context=_context("usr_1"),
            resolved_policy=policy, detected_needs=[],
        )
        assert result[0].resolved_date is None
        assert result[0].date_certainty is None
        assert result[0].date_resolution_status == status
    assert provider.purposes == ["applicability_relevance_card"] * 2


@pytest.mark.asyncio
async def test_persisted_date_survives_enforced_relevance_skip() -> None:
    from tests.applicability_support import _candidate as applicability_candidate
    from tests.memory.test_applicability_scorer_fallbacks import _plan

    provider = DateProvider()
    scorer = ApplicabilityScorer(
        LLMClient(provider_name=provider.name, providers=[provider]),
        FrozenClock(datetime(2026, 3, 30, tzinfo=timezone.utc)), _settings(),
    )
    row = applicability_candidate("memory_1", canonical_text=_TEXT)
    row["payload_json"] = {
        "source_message_ids": ["source_1"],
        "source_occurred_at": _SOURCE_DATE,
        "date_resolution": _annotation("exact|ref|days|-2"),
    }
    row["evidence_packets"] = [{"support_kind": "direct", "spans": [{"quote_text": _TEXT}]}]
    plan = _plan().model_copy(update={
        "answer_shape": "single_fact", "source_precision": "required", "exact_recall_mode": True,
    })
    policy = PolicyResolver().resolve(
        ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"], None, None,
    )
    result = await scorer.score_shortlist(
        [row], message_text="When did I sign the lease?",
        conversation_context=_context("usr_1"), resolved_policy=policy,
        detected_needs=[], retrieval_plan=plan, applicability_gate_mode="enforced",
    )
    assert provider.purposes == []
    assert result[0].resolved_date == "2025-03-08"
    assert result[0].date_certainty == "exact"
    assert result[0].date_resolution_status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", [
    "range", "start_only", "end_only", "calendar_month",
    "uncertain_day", "no_offset", "event_no_offset", "pending_end", "stale", "mismatched_bounds",
])
async def test_sqlite_interval_context_preserves_bounds_instead_of_terminal_point(variant: str) -> None:
    from atagia.memory.extraction_temporal import (
        ResolvedTemporalEndpoint, TemporalEndpoint, TemporalIntervalResolution,
        resolved_endpoint_timestamp,
    )

    # These are the existing shared-year range and whole-month acceptance inputs.
    text = "The user is away from June 10 through June 12, 2026, inclusive."
    reference = "2026-09-27T12:00:00+00:00"
    if variant == "calendar_month":
        text = (
            "The user is on leave for the whole of November 2026, "
            "from the month's first instant through its last instant, UTC+00:00."
        )

    def annotation(resolution_text: str, output: str) -> DateResolution:
        return DateResolution(
            **parse_date_resolution(output, reference).model_dump(),
            source_text_sha256=sha256(resolution_text.encode("utf-8")).hexdigest(),
            reference_date="2026-09-27",
        )

    if variant == "event_no_offset":
        text = "The user visited Jo yesterday."
    point = annotation(
        text, "analyze" if variant == "calendar_month"
        else "exact|ref|days|-1" if variant == "event_no_offset"
        else "exact|2026-06-12|days|0",
    )
    endpoints = []
    for day in (10, 12):
        if variant == "calendar_month":
            descriptor = TemporalEndpoint(
                text=None, source_quote="the whole of November 2026",
                time=None, offset="+00:00", period="month",
                calendar_period={"year": 2026, "month": 11},
            )
            resolution = None
        elif variant == "event_no_offset":
            descriptor = TemporalEndpoint(text=None, time=None, offset=None)
            resolution = point
        else:
            endpoint_text = f"June {day}, 2026"
            descriptor = TemporalEndpoint(
                text=endpoint_text, source_quote=f"June {day}", time=None,
                offset=None if variant == "no_offset" else "+00:00",
            )
            certainty = "uncertain" if variant == "uncertain_day" else "exact"
            resolution = annotation(
                endpoint_text, "analyze" if variant == "pending_end" and day == 12
                else f"{certainty}|2026-06-{day}|days|0",
            )
        endpoints.append(ResolvedTemporalEndpoint(
            endpoint=descriptor, date_resolution=resolution,
            source_message_id="source_1", source_occurred_at=reference,
        ))
    interval = TemporalIntervalResolution(
        start=None if variant == "end_only" else endpoints[0],
        end=None if variant == "start_only" else endpoints[1],
    )
    start = resolved_endpoint_timestamp(interval.start, is_end=False)
    end = resolved_endpoint_timestamp(interval.end, is_end=True)
    if variant == "mismatched_bounds":
        start = "2026-06-11T00:00:00+00:00"
    payload = {
        "source_message_ids": ["source_1"], "source_occurred_at": reference,
        "date_resolution": point.model_dump(mode="json"),
        "date_interval": interval.model_dump(mode="json"),
    }
    if variant == "stale":
        text = "The user is there through Sunday."
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        clock = FrozenClock(datetime(2026, 9, 28, tzinfo=timezone.utc))
        await UserRepository(connection, clock).create_user("usr_1")
        memories = MemoryObjectRepository(connection, clock)
        await memories.create_memory_object(
            user_id="usr_1", object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER, canonical_text=text,
            source_kind=MemorySourceKind.EXTRACTED, confidence=0.9,
            privacy_level=0, memory_id="memory_interval",
            temporal_type="event_triggered" if variant == "event_no_offset" else "bounded",
            valid_from=start, valid_to=end, payload=payload,
        )
        row = await memories.get_memory_object("memory_interval", "usr_1")
        assert row is not None
        row["evidence_packets"] = [{
            "support_kind": "direct", "spans": [{"span_role": "source", "quote_text": text}],
        }]
        provider = DateProvider()
        policy = PolicyResolver().resolve(
            ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"], None, None,
        )
        scored = await ApplicabilityScorer(
            LLMClient(provider_name=provider.name, providers=[provider]), clock, _settings(),
        ).score_shortlist(
            [_candidate(row)], message_text="When was it?", conversation_context=_context("usr_1"),
            resolved_policy=policy, detected_needs=[],
        )
        context = ContextComposer(clock).compose(
            scored_candidates=scored, current_contract={}, user_state=None,
            resolved_policy=policy, conversation_messages=[], query_type="temporal",
            enable_final_answer_evidence_pack=True,
        )
        item = context.answer_evidence_items[0]
        assert provider.purposes == ["applicability_relevance_card"]
        assert item["date_kind"] == ("point" if variant == "event_no_offset" else "interval")
        if variant == "event_no_offset":
            assert item["date"] == "2026-09-26"
            assert item["date_certainty"] == "exact"
            assert item["date_resolution_status"] == "completed"
            assert item["normalization"]["resolved_date"] == "2026-09-26"
        elif variant == "pending_end":
            assert item["date"] == "from 2026-06-10T00:00:00+00:00; end not resolved"
            assert item["date_certainty"] is None
            assert item["date_resolution_status"] == "pending_analysis"
        elif variant in {"uncertain_day", "no_offset", "stale", "mismatched_bounds"}:
            assert item["date"] == ""
            if variant in {"stale", "mismatched_bounds"}:
                assert item["date_resolution_status"] == "stale"
                assert "valid_from" not in item["normalization"]
            elif variant == "uncertain_day":
                assert item["date_certainty"] == "uncertain"
        else:
            expected = (
                f"from {start} through {end}" if start and end
                else f"from {start}" if start else f"through {end}"
            )
            assert item["date"] == expected
            assert f"- date: {expected}" in context.answer_evidence_block
            assert item["date_certainty"] == "exact"
            assert item["date_resolution_status"] == "completed"
            assert item["normalization"]["date_interval"] == payload["date_interval"]
            assert "resolved_date" not in item["normalization"]
        if variant == "calendar_month":
            assert start == "2026-11-01T00:00:00+00:00"
            assert end == "2026-11-30T23:59:59+00:00"
            assert item["normalization"]["point_date_resolution"]["status"] == "pending_analysis"
            assert "- date_resolution_status: completed" in context.answer_evidence_block
    finally:
        await close_connection(connection)
