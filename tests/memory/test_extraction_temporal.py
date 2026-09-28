"""Offline checks for single-candidate temporal decisions."""

from __future__ import annotations

import asyncio
import html
import json
from types import SimpleNamespace

import pytest

from atagia.memory import extraction_cards
from atagia.memory.extraction_cards import CandidateDraft, CardResult, run_temporal_cards
from atagia.memory.extraction_temporal import (
    build_temporal_interval_prompt,
    build_temporal_type_question,
    parse_temporal_interval_output,
    parse_temporal_type_output,
    ResolvedTemporalEndpoint,
    resolved_endpoint_timestamp,
)
from atagia.memory.date_resolution import DateResolution, parse_date_resolution
from atagia.models.schemas_memory import ExtractionConversationContext, ExtractionContextMessage, LeanTemporalStatus
from atagia.services.llm_client import LLMClient, LLMError


class ScriptedTemporalClient:
    def __init__(self, answers: dict[tuple[str, str], str]) -> None:
        self.answers = answers
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        purpose = request.metadata["purpose"]
        if request.choice_questions is not None:
            return SimpleNamespace(choice_answers={
                candidate_id: SimpleNamespace(choice=self.answers[(purpose, candidate_id)])
                for candidate_id in request.choice_questions
            })
        prompt = request.messages[-1].content
        candidate_id = request.metadata.get("memory_candidate_id") or request.metadata.get("stage") or next(
            candidate_id for candidate_id in ("cand_001", "cand_002", "cand_003")
            if f"{candidate_id}: " in prompt
        )
        return SimpleNamespace(output_text=self.answers[(purpose, candidate_id)])

    async def complete_choice_questions(self, **kwargs):
        return await LLMClient.complete_choice_questions(self, **kwargs)


def _context() -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_temporal",
        conversation_id="cnv_temporal",
        source_message_id="msg_temporal",
        workspace_id="ws_temporal",
        assistant_mode_id="coding_debug",
        recent_messages=[],
        privacy_enforcement="off",
    )


def test_temporal_type_requires_one_explicit_answer() -> None:
    assert parse_temporal_type_output("none") is None
    assert parse_temporal_type_output("unknown") == "unknown"
    assert parse_temporal_type_output("event_triggered") == "event_triggered"
    for output in ("", "cand_001 event_triggered", "bounded 2026-09-26", "temporary"):
        with pytest.raises(ValueError, match="Invalid temporal type"):
            parse_temporal_type_output(output)


def _endpoint(
    *, text=None, source_quote=None, time=None, offset="-04:00", period="day", calendar_period=None,
) -> str:
    return json.dumps({
        "text": text, "source_quote": source_quote, "time": time,
        "offset": offset, "period": period, "calendar_period": calendar_period,
    })


def _date_result(output="exact|ref|days|-1") -> DateResolution:
    return DateResolution(
        **parse_date_resolution(output, "2026-09-26").model_dump(),
        source_text_sha256="0" * 64, reference_date="2026-09-26",
    )


def test_temporal_interval_validates_format_offsets_and_order() -> None:
    start, end = parse_temporal_interval_output(
        _endpoint(time="10:00:00", offset="+02:00") + "\n" + _endpoint(time="11:00:00", offset="+02:00")
    )
    assert resolved_endpoint_timestamp(
        ResolvedTemporalEndpoint(endpoint=start, date_resolution=_date_result()), is_end=False,
    ) == "2026-09-25T10:00:00+02:00"
    assert resolved_endpoint_timestamp(
        ResolvedTemporalEndpoint(endpoint=end, date_resolution=_date_result()), is_end=True,
    ) == "2026-09-25T11:00:00+02:00"
    assert parse_temporal_interval_output("null\nnull") == (None, None)
    for output in (
        "null", "2026-09-26T10:00:00\nnull",
        _endpoint(time="25:00:00") + "\nnull",
        _endpoint(offset="+25:00") + "\nnull",
        _endpoint(text="June 10, 2026") + "\nnull",
        _endpoint(period="month", source_quote="November 2026", calendar_period={"year": 2026, "month": None}) + "\nnull",
        _endpoint(period="day", source_quote="November 2026", calendar_period={"year": 2026, "month": 11}) + "\nnull",
        _endpoint(period="year", source_quote="November 2026", calendar_period={"year": 2026, "month": 11}) + "\nnull",
        "null\nnull\nexplanation: none", 'start: null\nend: null',
        '{"text": null, "time": null, "offset": "Z", "extra": true}\nnull',
    ):
        with pytest.raises(ValueError):
            parse_temporal_interval_output(output)


@pytest.mark.parametrize("output,offset", [
    ("uncertain|ref|days|-1", "-04:00"), ("unknown", "-04:00"),
    ("analyze", "-04:00"), ("exact|ref|days|-1", None),
])
def test_interval_never_invents_an_exact_bound(output, offset) -> None:
    endpoint, _ = parse_temporal_interval_output(_endpoint(offset=offset) + "\nnull")
    assert resolved_endpoint_timestamp(
        ResolvedTemporalEndpoint(endpoint=endpoint, date_resolution=_date_result(output)), is_end=False,
    ) is None


def test_temporal_interval_prompt_requires_a_relevant_type() -> None:
    with pytest.raises(ValueError, match="does not need an interval"):
        build_temporal_interval_prompt(
            candidate_id="cand_001",
            candidate_text="The user likes jazz.",
            temporal_type="permanent",
            source_context="<message_timestamp>none</message_timestamp>",
        )


@pytest.mark.asyncio
async def test_temporal_pipeline_resolves_only_relevant_intervals() -> None:
    client = ScriptedTemporalClient(
        {
            ("memory_extraction_temporal_type_card", "cand_001"): "event_triggered",
            ("memory_extraction_temporal_interval_card", "cand_001"): (
                _endpoint() + "\n" + _endpoint()
            ),
            ("memory_date_resolution", "cand_001"): "exact|ref|days|-1",
            ("memory_date_resolution", "cand_002"): "unknown",
            ("memory_extraction_temporal_type_card", "cand_002"): "permanent",
            ("memory_extraction_temporal_type_card", "cand_003"): "none",
        }
    )
    result = await run_temporal_cards(
        llm_client=client,
        model="openrouter/openai/gpt-6-luna",
        date_model="openrouter/openai/gpt-6-luna,low",
        temporal_type_model="openrouter/openai/gpt-6-luna",
        message_text="Yesterday I met Jo. I like jazz. The code is AX-4.",
        role="user",
        context=_context(),
        occurred_at="2026-09-26T12:00:00-04:00",
        prior_chunk_context="Earlier discussion of Jo.",
        candidates=(
            CandidateDraft("cand_001", "The user met Jo yesterday."),
            CandidateDraft("cand_002", "The user likes jazz."),
            CandidateDraft("cand_003", "The code is AX-4."),
        ),
        metadata={"test_tag": "temporal"},
        semaphore=asyncio.Semaphore(2),
    )

    assert result.parsed["cand_001"].type == "event_triggered"
    assert result.parsed["cand_001"].valid_from_iso == "2026-09-25T00:00:00-04:00"
    assert result.parsed["cand_002"].type == "permanent"
    assert result.parsed["cand_002"].valid_from_iso is None
    assert result.parsed["cand_003"].date_not_applicable
    assert result.parsed["cand_001"].date_resolution.resolved_date == "2026-09-25"
    assert result.parsed["cand_002"].date_resolution.certainty == "unknown"
    assert len(client.requests) == 6
    assert [request.metadata["purpose"] for request in client.requests].count(
        "memory_extraction_temporal_interval_card"
    ) == 1
    assert all(
        request.finite_choice and request.metadata["stage"] in {
            "cand_001", "cand_002", "cand_003"
        }
        for request in client.requests
        if request.metadata["purpose"] == "memory_extraction_temporal_type_card"
    )
    assert all(
        request.model == ("openrouter/openai/gpt-6-luna,low" if request.metadata["purpose"] == "memory_date_resolution" else "openrouter/openai/gpt-6-luna")
        for request in client.requests
    )
    interval_prompt = next(
        request.messages[-1].content
        for request in client.requests
        if request.metadata["purpose"] == "memory_extraction_temporal_interval_card"
    )
    assert "<message_timestamp>2026-09-26T12:00:00-04:00" in interval_prompt
    assert "Earlier discussion of Jo." in interval_prompt


@pytest.mark.asyncio
async def test_typesafe_batches_types_but_interval_keeps_extractor_model() -> None:
    client = ScriptedTemporalClient({
        ("memory_extraction_temporal_type_card", "cand_001"): "bounded",
        ("memory_extraction_temporal_type_card", "cand_002"): "unknown",
        ("memory_date_resolution", "cand_001"): "exact|ref|days|+1",
        ("memory_date_resolution", "cand_002"): "unknown",
        ("memory_extraction_temporal_interval_card", "cand_001"): (
            "null\n" + _endpoint(time="17:00:00")
        ),
    })
    result = await asyncio.wait_for(
        run_temporal_cards(
            llm_client=client,
            model="openrouter/openai/gpt-6-luna",
            date_model="openrouter/openai/gpt-6-luna,low",
            temporal_type_model="typesafe/jev-1.13.0",
            message_text="I am there through Sunday; the rest has unclear timing.",
            role="user",
            context=_context(),
            occurred_at="2026-09-26T12:00:00-04:00",
            prior_chunk_context=None,
            candidates=(
                CandidateDraft("cand_001", "The user is there through Sunday."),
                CandidateDraft("cand_002", "The other state has unclear timing."),
            ),
            metadata={},
            semaphore=asyncio.Semaphore(1),
        ),
        timeout=2,
    )
    assert len(client.requests) == 4
    type_request = next(request for request in client.requests if request.metadata["purpose"] == "memory_extraction_temporal_type_card")
    interval_request = next(request for request in client.requests if request.metadata["purpose"] == "memory_extraction_temporal_interval_card")
    assert type_request.model == "typesafe/jev-1.13.0"
    assert set(type_request.choice_questions) == {"cand_001", "cand_002"}
    assert type_request.metadata["purpose"] == "memory_extraction_temporal_type_card"
    assert "<message_timestamp>2026-09-26T12:00:00-04:00" in type_request.messages[0].content
    assert interval_request.model == "openrouter/openai/gpt-6-luna"
    assert interval_request.choice_questions is None
    assert interval_request.metadata["purpose"] == "memory_extraction_temporal_interval_card"
    assert result.parsed["cand_001"].valid_from_iso is None
    assert result.parsed["cand_001"].valid_to_iso == "2026-09-27T17:00:00-04:00"
    assert result.parsed["cand_002"].type == "unknown"
    assert result.parsed["cand_002"].valid_from_iso is None


@pytest.mark.asyncio
async def test_llm_interval_starts_after_its_own_type_finishes() -> None:
    interval_started = asyncio.Event()

    class Client:
        async def complete(self, request):
            if request.metadata["purpose"] == "memory_extraction_temporal_interval_card":
                interval_started.set()
                return SimpleNamespace(output_text="null\nnull")
            if request.metadata["stage"] == "cand_002":
                await interval_started.wait()
                return SimpleNamespace(output_text="permanent")
            return SimpleNamespace(output_text="event_triggered")

        async def complete_choice_questions(self, **kwargs):
            return await LLMClient.complete_choice_questions(self, **kwargs)

    result = await asyncio.wait_for(
        run_temporal_cards(
            llm_client=Client(),
            model="openrouter/openai/gpt-6-luna",
            date_model="openrouter/openai/gpt-6-luna,low",
            temporal_type_model="openrouter/openai/gpt-6-luna",
            message_text="I visited Jo. I like jazz.",
            role="user",
            context=_context(),
            occurred_at=None,
            prior_chunk_context=None,
            candidates=(
                CandidateDraft("cand_001", "The user visited Jo."),
                CandidateDraft("cand_002", "The user likes jazz."),
            ),
            metadata={},
            semaphore=asyncio.Semaphore(2),
        ),
        timeout=2,
    )
    assert interval_started.is_set()
    assert result.parsed["cand_001"].date_resolution.status == "pending_analysis"
    assert result.parsed["cand_001"].date_resolution.reference_date is None
    assert result.parsed["cand_001"].type == "event_triggered"
    assert result.parsed["cand_002"].type == "permanent"


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", [False, True])
async def test_copied_interval_text_keeps_its_source_anchor(ambiguous: bool) -> None:
    class Client(ScriptedTemporalClient):
        async def complete(self, request):
            if request.metadata["purpose"] == "memory_date_resolution":
                self.requests.append(request)
                return SimpleNamespace(output_text="exact|ref|days|-1")
            return await super().complete(request)

    client = Client({
        ("memory_extraction_temporal_type_card", "cand_001"): "event_triggered",
        ("memory_extraction_temporal_interval_card", "cand_001"): _endpoint(text="yesterday", source_quote="yesterday") + "\nnull",
    })
    context = _context().model_copy(update={"recent_messages": [ExtractionContextMessage(
        id="msg_prior", role="user", content="The user visited Jo yesterday.",
        occurred_at="2026-09-25T12:00:00-04:00",
    )]})
    candidate = "The user visited Jo yesterday." if ambiguous else "The user is there through Sunday."
    result = await run_temporal_cards(
        llm_client=client, model="openrouter/openai/gpt-6-luna",
        temporal_type_model="openrouter/openai/gpt-6-luna",
        date_model="openrouter/openai/gpt-6-luna,low",
        message_text=candidate, role="user", context=context,
        occurred_at="2026-09-26T12:00:00-04:00", prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", candidate),), metadata={}, semaphore=asyncio.Semaphore(2),
    )
    status = result.parsed["cand_001"]
    start = status.date_interval.start
    assert start is not None
    if ambiguous:
        assert start.date_resolution.status == "pending_analysis"
        assert start.source_message_id is None
        assert status.valid_from_iso is None
    else:
        assert start.date_resolution.reference_date == "2026-09-25"
        assert start.source_message_id == "msg_prior"
        assert status.valid_from_iso == "2026-09-24T00:00:00-04:00"
    date_requests = [request for request in client.requests if request.metadata["purpose"] == "memory_date_resolution"]
    assert len(date_requests) == (1 if ambiguous else 2)


@pytest.mark.asyncio
async def test_typed_written_period_keeps_ambiguous_source_pending() -> None:
    # The unchanged original whole_month source is present under two source
    # identities. Typed calendar fields must not bypass unresolved attribution.
    text = "The user is on leave for the whole of November 2026, from the month's first instant through its last instant, UTC+00:00."
    endpoint = _endpoint(
        source_quote=text, offset="+00:00", period="month",
        calendar_period={"year": 2026, "month": 11},
    )
    client = ScriptedTemporalClient({
        ("memory_extraction_temporal_type_card", "cand_001"): "bounded",
        ("memory_extraction_temporal_interval_card", "cand_001"): endpoint + "\n" + endpoint,
        ("memory_date_resolution", "cand_001"): "analyze",
    })
    context = _context().model_copy(update={"recent_messages": [ExtractionContextMessage(
        id="msg_prior", role="user", content=text, occurred_at="2026-09-26T12:00:00+00:00",
    )]})
    result = await run_temporal_cards(
        llm_client=client, model="openrouter/openai/gpt-6-luna",
        temporal_type_model="openrouter/openai/gpt-6-luna",
        date_model="openrouter/openai/gpt-6-luna,low",
        message_text=text, role="user", context=context,
        occurred_at="2026-09-27T12:00:00+00:00", prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", text),), metadata={}, semaphore=asyncio.Semaphore(2),
    )
    status = result.parsed["cand_001"]
    assert status.valid_from_iso is None and status.valid_to_iso is None
    assert status.date_interval.start.date_resolution.status == "pending_analysis"
    assert status.date_interval.end.date_resolution.status == "pending_analysis"
    assert status.date_interval.start.source_message_id is None
    assert len([request for request in client.requests if request.metadata["purpose"] == "memory_date_resolution"]) == 1


_WHOLE_MONTH_SOURCE = (
    "The user is on leave for the whole of November 2026, from the month's "
    "first instant through its last instant, UTC+00:00."
)


async def _run_month_quote(
    source: str, quote: str, recent_messages: tuple[ExtractionContextMessage, ...] = (),
) -> LeanTemporalStatus:
    endpoint = _endpoint(
        source_quote=quote, offset="+00:00", period="month",
        calendar_period={"year": 2026, "month": 11},
    )
    client = ScriptedTemporalClient({
        ("memory_extraction_temporal_type_card", "cand_001"): "bounded",
        ("memory_extraction_temporal_interval_card", "cand_001"): endpoint + "\n" + endpoint,
        ("memory_date_resolution", "cand_001"): "analyze",
    })
    result = await run_temporal_cards(
        llm_client=client, model="openrouter/openai/gpt-6-luna",
        temporal_type_model="openrouter/openai/gpt-6-luna",
        date_model="openrouter/openai/gpt-6-luna,low",
        message_text=source, role="user",
        context=_context().model_copy(update={"recent_messages": list(recent_messages)}),
        occurred_at="2026-09-27T12:00:00+00:00", prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", _WHOLE_MONTH_SOURCE),),
        metadata={}, semaphore=asyncio.Semaphore(2),
    )
    return result.parsed["cand_001"]


@pytest.mark.asyncio
@pytest.mark.parametrize("source_encoding", [0, 1, 2], ids=["apostrophe", "literal_entity", "nested_entity"])
@pytest.mark.parametrize("escaped_quote", [False, True], ids=["raw_quote", "prompt_encoded_quote"])
async def test_interval_quote_encoding_preserves_original_source(
    source_encoding: int, escaped_quote: bool,
) -> None:
    # Mechanical encoding variants of the original whole_month input; the
    # calendar fields and expected interval remain unchanged.
    source = _WHOLE_MONTH_SOURCE
    for _ in range(source_encoding):
        source = html.escape(source)
    quote = html.escape(source) if escaped_quote else source
    status = await _run_month_quote(source, quote)
    assert status.valid_from_iso == "2026-11-01T00:00:00+00:00"
    assert status.valid_to_iso == "2026-11-30T23:59:59+00:00"
    for endpoint in (status.date_interval.start, status.date_interval.end):
        assert endpoint.endpoint.source_quote == source
        assert endpoint.source_message_id == "msg_temporal"
        assert endpoint.source_occurred_at == "2026-09-27T12:00:00+00:00"
        assert endpoint.date_resolution is None


@pytest.mark.asyncio
async def test_prompt_encoded_interval_quote_keeps_recent_source_identity() -> None:
    recent = ExtractionContextMessage(
        id="msg_prior", role="user", content=_WHOLE_MONTH_SOURCE,
        occurred_at="2026-09-26T12:00:00+00:00",
    )
    status = await _run_month_quote(
        "The user is away from June 10 through June 12, 2026, inclusive.",
        html.escape(_WHOLE_MONTH_SOURCE), (recent,),
    )
    endpoint = status.date_interval.start
    assert endpoint.endpoint.source_quote == _WHOLE_MONTH_SOURCE
    assert endpoint.source_message_id == "msg_prior"
    assert endpoint.source_occurred_at == recent.occurred_at
    assert status.valid_from_iso == "2026-11-01T00:00:00+00:00"


@pytest.mark.asyncio
@pytest.mark.parametrize("quote", [
    html.escape(html.escape(_WHOLE_MONTH_SOURCE)),
    html.escape(_WHOLE_MONTH_SOURCE + " invented"),
], ids=["requires_two_decodings", "invented_quote"])
async def test_interval_quote_encoding_does_not_repair_unmatched_source(quote: str) -> None:
    with pytest.raises(ValueError, match="source"):
        await _run_month_quote(_WHOLE_MONTH_SOURCE, quote)


@pytest.mark.asyncio
@pytest.mark.parametrize("same_message", [False, True], ids=["different_sources", "different_spans"])
async def test_interval_quote_encoding_keeps_ambiguous_matches_pending(same_message: bool) -> None:
    encoded = html.escape(_WHOLE_MONTH_SOURCE)
    # The returned quote matches one original span literally and another span
    # only as displayed in the prompt. Neither representation has precedence.
    if same_message:
        source = _WHOLE_MONTH_SOURCE + " " + encoded
        recent = ()
    else:
        source = _WHOLE_MONTH_SOURCE
        recent = (ExtractionContextMessage(
            id="msg_prior", role="user", content=encoded,
            occurred_at="2026-09-26T12:00:00+00:00",
        ),)
    status = await _run_month_quote(source, encoded, recent)
    assert status.valid_from_iso is None and status.valid_to_iso is None
    for endpoint in (status.date_interval.start, status.date_interval.end):
        assert endpoint.endpoint.source_quote == encoded
        assert endpoint.date_resolution.status == "pending_analysis"
        assert endpoint.source_message_id is None
        assert endpoint.source_occurred_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("synthesized_origin", ["candidate", "prior_chunk_summary"])
async def test_interval_source_quote_cannot_use_synthesized_text(synthesized_origin: str) -> None:
    source = "The user is away from June 10 through June 12, 2026, inclusive."
    normalized = "The user is away from June 10, 2026 through June 12, 2026."
    candidate = normalized if synthesized_origin == "candidate" else source
    prior_summary = normalized if synthesized_origin == "prior_chunk_summary" else None
    client = ScriptedTemporalClient({
        ("memory_extraction_temporal_type_card", "cand_001"): "bounded",
        ("memory_extraction_temporal_interval_card", "cand_001"): _endpoint(
            text="June 10, 2026", source_quote="June 10, 2026", offset="+02:00",
        ) + "\nnull",
        ("memory_date_resolution", "cand_001"): "exact|2026-06-12|days|0",
    })
    with pytest.raises(ValueError, match="source"):
        await run_temporal_cards(
            llm_client=client, model="openrouter/openai/gpt-6-luna",
            temporal_type_model="openrouter/openai/gpt-6-luna",
            date_model="openrouter/openai/gpt-6-luna,low",
            message_text=source, role="user", context=_context(),
            occurred_at="2026-06-01T12:00:00+02:00", prior_chunk_context=prior_summary,
            candidates=(CandidateDraft("cand_001", candidate),), metadata={}, semaphore=asyncio.Semaphore(2),
        )


@pytest.mark.asyncio
async def test_temporal_pipeline_does_not_turn_malformed_type_into_unknown() -> None:
    client = ScriptedTemporalClient(
        {("memory_extraction_temporal_type_card", "cand_001"): "temporary"}
    )
    with pytest.raises(LLMError, match="unknown option"):
        await run_temporal_cards(
            llm_client=client,
            model="openrouter/openai/gpt-6-luna",
            date_model="openrouter/openai/gpt-6-luna,low",
            temporal_type_model="openrouter/openai/gpt-6-luna",
            message_text="I am at the station.",
            role="user",
            context=_context(),
            occurred_at="2026-09-26T12:00:00-04:00",
            prior_chunk_context=None,
            candidates=(CandidateDraft("cand_001", "The user is at the station."),),
            metadata={},
            semaphore=asyncio.Semaphore(2),
        )
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_temporal_pipeline_cancels_sibling_after_technical_failure() -> None:
    class FailingClient:
        def __init__(self) -> None:
            self.sibling_started = asyncio.Event()
            self.sibling_cancelled = asyncio.Event()

        async def complete(self, request):
            if request.metadata["stage"] == "cand_002":
                self.sibling_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.sibling_cancelled.set()
                    raise
            await self.sibling_started.wait()
            raise ValueError("Provider failed")

        async def complete_choice_questions(self, **kwargs):
            return await LLMClient.complete_choice_questions(self, **kwargs)

    client = FailingClient()
    with pytest.raises(ValueError, match="Provider failed"):
        await asyncio.wait_for(
            run_temporal_cards(
                llm_client=client,
                model="openrouter/openai/gpt-6-luna",
                date_model="openrouter/openai/gpt-6-luna,low",
                temporal_type_model="openrouter/openai/gpt-6-luna",
                message_text="I visited Jo and met May.",
                role="user",
                context=_context(),
                occurred_at="2026-09-26T12:00:00-04:00",
                prior_chunk_context=None,
                candidates=(
                    CandidateDraft("cand_001", "The user visited Jo."),
                    CandidateDraft("cand_002", "The user met May."),
                ),
                metadata={},
                semaphore=asyncio.Semaphore(2),
            ),
            timeout=2,
        )
    assert client.sibling_cancelled.is_set()


@pytest.mark.asyncio
async def test_temporal_failure_cancels_other_enrichment_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_started = asyncio.Event()
    evidence_cancelled = asyncio.Event()

    async def fake_card(*, card_name, **kwargs):
        del kwargs
        if card_name == "candidate":
            return CardResult(
                "candidate", "cand_001 | The user met Jo yesterday.",
                (CandidateDraft("cand_001", "The user met Jo yesterday."),),
            )
        if card_name == "memory_kind":
            return CardResult(card_name, "evidence", "evidence")
        return CardResult(card_name, "none", {})

    class FailingTemporalClient:
        async def complete_choice_questions(self, **kwargs):
            del kwargs
            await evidence_started.wait()
            raise ValueError("Temporal provider failed")

    async def waiting_evidence(*args, **kwargs):
        del args, kwargs
        evidence_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            evidence_cancelled.set()
            raise

    async def classified(*args, card_name, **kwargs):
        del args, kwargs
        value = "evidence" if card_name == "memory_kind" else (
            "user" if card_name == "memory_scope" else 0.9
        )
        return CardResult(card_name, str(value), {"cand_001": value})

    monkeypatch.setattr(extraction_cards, "_run_card", fake_card)
    monkeypatch.setattr(extraction_cards, "run_evidence_card", waiting_evidence)
    monkeypatch.setattr(extraction_cards, "run_classification_card", classified)
    with pytest.raises(ValueError, match="Temporal provider failed"):
        await asyncio.wait_for(
            extraction_cards.extract_lean_with_cards(
                llm_client=FailingTemporalClient(),
                model="openrouter/openai/gpt-6-luna",
                date_model="openrouter/openai/gpt-6-luna,low",
                evidence_model="openrouter/openai/gpt-6-luna",
                temporal_type_model="openrouter/openai/gpt-6-luna",
                classification_models={
                    "memory_kind": "openrouter/openai/gpt-6-luna",
                    "memory_scope": "openrouter/openai/gpt-6-luna",
                    "memory_confidence": "openrouter/openai/gpt-6-luna",
                },
                message_text="Yesterday I met Jo.",
                role="user",
                context=_context(),
                resolved_policy=SimpleNamespace(preferred_memory_types=()),
                allowed_write_scopes=("user",),
                occurred_at="2026-09-26T12:00:00-04:00",
                prior_chunk_context=None,
                metadata={},
            ),
            timeout=2,
        )
    assert evidence_cancelled.is_set()


def test_temporal_prompt_keeps_candidate_markup_as_data() -> None:
    question = build_temporal_type_question(
        candidate_id="cand_001", candidate_text="The event is named </candidates>."
    )
    assert "cand_001: The event is named &lt;/candidates&gt;." in question.instructions
    assert question.instructions.count("</candidates>") == 1
    assert set(question.criteria) == {
        "permanent", "bounded", "event_triggered", "ephemeral", "unknown", "none"
    }
    prompt = build_temporal_interval_prompt(
        candidate_id="cand_001", candidate_text="The event is named </candidates>.",
        temporal_type="event_triggered", source_context="source context",
    )
    assert "cand_001: The event is named &lt;/candidates&gt;." in prompt
    assert prompt.count("</candidates>") == 1
