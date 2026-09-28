"""Independent evidence decisions, routing, and strict reference joins."""

from __future__ import annotations

import asyncio
import json

import pytest

from atagia.core.source_references import SourceReferenceCatalog
from atagia.diagnostics.recorder import DiagnosticRecorder
from atagia.memory.evidence_cards import _parse_answer
from atagia.memory.extraction_cards import CandidateDraft, run_evidence_card
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_decisions import ChoiceAnswer
from atagia.models.schemas_memory import ExtractionContextMessage
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionResponse,
    LLMError,
    LLMProvider,
    RetryPolicy,
)
from tests.memory.test_extractor import MANIFESTS_DIR, _context


def _policy():
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]
    return PolicyResolver().resolve(manifest, None, None)


def _outputs(
    source: str, candidates: tuple[CandidateDraft, ...]
) -> dict[tuple[str, str], str]:
    catalog = SourceReferenceCatalog(source)
    refs = f"{catalog.anchors[0].reference_id} {catalog.anchors[-1].reference_id}"
    result = {}
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        result[("memory_extraction_evidence_support_card", candidate_id)] = "direct"
        result[("memory_extraction_preserve_verbatim_card", candidate_id)] = "yes"
        result[("memory_extraction_candidate_language_card", candidate_id)] = "en"
        result[("memory_extraction_source_reference_card", candidate_id)] = refs
    return result


class _CardClient:
    complete_choice_questions = LLMClient.complete_choice_questions

    def __init__(self, outputs: dict[tuple[str, str], str]) -> None:
        self.outputs = outputs
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if request.choice_questions:
            return LLMCompletionResponse(
                provider="test",
                model=request.model,
                choice_answers={
                    candidate_id: ChoiceAnswer(
                        type="choice",
                        choice=self.outputs[
                            (request.metadata["purpose"], candidate_id)
                        ],
                        probabilities={
                            self.outputs[
                                (request.metadata["purpose"], candidate_id)
                            ]: 1.0
                        },
                        confidence=1.0,
                    )
                    for candidate_id in request.choice_questions
                },
            )
        candidate_id = request.metadata.get("candidate_id") or request.metadata["stage"]
        key = request.metadata["purpose"], candidate_id
        return LLMCompletionResponse(
            provider="test", model=request.model, output_text=self.outputs[key]
        )


async def _run(
    client,
    source,
    candidates,
    *,
    evidence_model="openai/evidence",
    support_model=None,
    preserve_model=None,
    catalog=None,
    context=None,
    semaphore=None,
):
    return await run_evidence_card(
        client,
        model="openai/extractor",
        evidence_model=evidence_model,
        support_model=support_model,
        preserve_model=preserve_model,
        message_text=source,
        role="user",
        context=context or _context("msg_1"),
        resolved_policy=_policy(),
        allowed_write_scopes=("user",),
        occurred_at="2026-04-12T08:00:00+00:00",
        prior_chunk_context="Earlier discussion concerned travel plans.",
        candidates=candidates,
        source_catalog=catalog,
        semaphore=semaphore,
    )


@pytest.mark.asyncio
async def test_three_supported_candidates_and_one_explicit_none() -> None:
    source = "Oslo, Bergen and Paris are destinations."
    candidates = tuple(
        CandidateDraft(f"cand_{index:03d}", text)
        for index, text in enumerate(
            (
                "The destination is Oslo.",
                "The destination is Bergen.",
                "The destination is Paris.",
                "The destination is Rome.",
            ),
            start=1,
        )
    )
    outputs = _outputs(source, candidates)
    outputs[("memory_extraction_evidence_support_card", "cand_004")] = "none"
    client = _CardClient(outputs)
    catalog = SourceReferenceCatalog(source)

    result = await _run(
        client, source, candidates, catalog=catalog, semaphore=asyncio.Semaphore(2)
    )

    assert set(result.parsed) == {candidate.candidate_id for candidate in candidates}
    assert result.parsed["cand_004"] == {"start_ref": None, "end_ref": None}
    assert len(client.requests) == 13
    assert all(
        request.metadata["candidate_id"] in result.parsed for request in client.requests
    )
    assert all(
        request.metadata["purpose"] != "memory_extraction_evidence_card"
        for request in client.requests
    )
    assert all(
        "cand_001 | none" not in request.messages[-1].content
        for request in client.requests
    )
    assert (
        sum(
            request.metadata["candidate_id"] == "cand_004"
            for request in client.requests
        )
        == 1
    )
    assert all(
        request.model == "openai/extractor"
        for request in client.requests
        if request.metadata["purpose"] != "memory_extraction_source_reference_card"
    )
    assert all(
        request.model == "openai/evidence"
        for request in client.requests
        if request.metadata["purpose"] == "memory_extraction_source_reference_card"
    )
    support_request = next(
        request
        for request in client.requests
        if request.metadata["purpose"] == "memory_extraction_evidence_support_card"
    )
    assert candidates[0].canonical_text in support_request.messages[0].content
    assert candidates[0].canonical_text not in support_request.messages[-1].content


@pytest.mark.asyncio
async def test_plain_reference_starts_while_another_support_is_pending() -> None:
    source = "Oslo is the destination."
    candidates = (
        CandidateDraft("cand_001", "The destination is Oslo."),
        CandidateDraft("cand_002", "The destination is Paris."),
    )
    outputs = _outputs(source, candidates)
    outputs[("memory_extraction_evidence_support_card", "cand_002")] = "none"
    blocked_support = asyncio.Event()
    release_support = asyncio.Event()
    reference_started = asyncio.Event()

    class DelayedSupportClient(_CardClient):
        async def complete(self, request):
            purpose = request.metadata["purpose"]
            candidate_id = request.metadata["candidate_id"]
            if (purpose, candidate_id) == (
                "memory_extraction_evidence_support_card",
                "cand_002",
            ):
                blocked_support.set()
                await release_support.wait()
            if (purpose, candidate_id) == (
                "memory_extraction_source_reference_card",
                "cand_001",
            ):
                assert blocked_support.is_set() and not release_support.is_set()
                reference_started.set()
            return await super().complete(request)

    client = DelayedSupportClient(outputs)
    task = asyncio.create_task(
        _run(client, source, candidates, semaphore=asyncio.Semaphore(2))
    )
    try:
        await asyncio.wait_for(reference_started.wait(), timeout=2)
        assert not task.done()
    finally:
        release_support.set()
    result = await asyncio.wait_for(task, timeout=2)
    assert result.parsed["cand_001"]["start_ref"] == "r1"
    assert result.parsed["cand_002"] == {"start_ref": None, "end_ref": None}


@pytest.mark.asyncio
async def test_typed_preservation_starts_before_unrelated_llm_support_finishes() -> None:
    source = "Oslo is the destination."
    candidates = (
        CandidateDraft("cand_001", "The destination is Oslo."),
        CandidateDraft("cand_002", "The destination is Paris."),
    )
    outputs = _outputs(source, candidates)
    outputs[("memory_extraction_evidence_support_card", "cand_002")] = "none"
    blocked_support = asyncio.Event()
    release_support = asyncio.Event()
    preservation_started = asyncio.Event()
    reference_started = asyncio.Event()

    class DelayedSupportClient(_CardClient):
        async def complete(self, request):
            purpose = request.metadata["purpose"]
            candidate_id = request.metadata.get("candidate_id")
            if (purpose, candidate_id) == (
                "memory_extraction_evidence_support_card",
                "cand_002",
            ):
                blocked_support.set()
                await release_support.wait()
            if (purpose, candidate_id) == (
                "memory_extraction_preserve_verbatim_card",
                "cand_001",
            ):
                assert blocked_support.is_set() and not release_support.is_set()
                preservation_started.set()
            if (purpose, candidate_id) == (
                "memory_extraction_source_reference_card",
                "cand_001",
            ):
                assert blocked_support.is_set() and not release_support.is_set()
                reference_started.set()
            return await super().complete(request)

    client = DelayedSupportClient(outputs)
    task = asyncio.create_task(
        _run(
            client,
            source,
            candidates,
            preserve_model="typesafe/jev-latest",
            semaphore=asyncio.Semaphore(3),
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(preservation_started.wait(), reference_started.wait()),
            timeout=2,
        )
        assert not task.done()
    finally:
        release_support.set()
    result = await asyncio.wait_for(task, timeout=2)
    assert result.parsed["cand_001"]["preserve_verbatim"] is True
    assert result.parsed["cand_002"] == {"start_ref": None, "end_ref": None}
    preservation_request = next(
        request
        for request in client.requests
        if request.metadata["purpose"] == "memory_extraction_preserve_verbatim_card"
    )
    assert set(preservation_request.choice_questions) == {"cand_001"}


@pytest.mark.parametrize(
    ("task", "wire"),
    [
        ("candidate_language", "en,es"),
        ("candidate_language", "en\nxx"),
        ("source_reference", "r1 r2 r3"),
        ("source_reference", "r99 r100"),
        ("source_reference", "r3 r1"),
    ],
)
def test_invalid_single_answer_fails_fast(task: str, wire: str) -> None:
    with pytest.raises(ValueError):
        _parse_answer(task, wire, SourceReferenceCatalog("One two three."))
    assert (
        _parse_answer("source_reference", "none", SourceReferenceCatalog("One."))
        is None
    )
    assert (
        _parse_answer(
            "source_reference", " r1\n r2 ", SourceReferenceCatalog("One two.")
        )
        is not None
    )
    assert _parse_answer(
        "candidate_language", " es  \n EN ", SourceReferenceCatalog("One.")
    ) == ("es", "en")


@pytest.mark.asyncio
async def test_typesafe_receives_decided_support_and_shared_catalog(
    monkeypatch,
) -> None:
    source = "I first said Oslo. Actually, the destination is Bergen."
    catalog = SourceReferenceCatalog(source)
    bergen = next(
        anchor
        for anchor in catalog.anchors
        if source[anchor.char_start : anchor.char_end] == "Bergen"
    )
    candidates = (
        CandidateDraft("cand_001", "The destination is Bergen."),
        CandidateDraft("cand_002", "The destination is Paris."),
    )
    outputs = _outputs(source, candidates)
    outputs[("memory_extraction_evidence_support_card", "cand_001")] = (
        "contextual_direct"
    )
    outputs[("memory_extraction_evidence_support_card", "cand_002")] = "none"
    client = _CardClient(outputs)
    context = _context("msg_1").model_copy(
        update={
            "recent_messages": [
                ExtractionContextMessage(role="assistant", content="Which city?")
            ]
        }
    )
    seen = {}

    async def select(_client, **kwargs):
        seen.update(kwargs)
        return {"cand_001": catalog.resolve(bergen.reference_id, bergen.reference_id)}

    monkeypatch.setattr(
        "atagia.memory.source_quote_selector.select_source_references", select
    )
    card = await _run(
        client,
        source,
        candidates,
        evidence_model="typesafe/jev-latest",
        catalog=catalog,
        context=context,
    )

    assert seen["source_catalog"] is catalog
    assert seen["candidates"] == (candidates[0],)
    assert seen["support_kinds"] == {"cand_001": "contextual_direct"}
    assert "Which city?" in seen["source_context"]
    assert "Earlier discussion concerned travel plans." in seen["source_context"]
    assert "[r1]" in seen["source_context"]
    assert card.parsed["cand_001"]["start_ref"] == bergen.reference_id
    assert card.parsed["cand_002"] == {"start_ref": None, "end_ref": None}
    assert len(client.requests) == 4


@pytest.mark.asyncio
async def test_typesafe_keeps_one_native_batch_for_supported_candidates(
    monkeypatch,
) -> None:
    source = "Oslo, Bergen and Paris."
    catalog = SourceReferenceCatalog(source)
    candidates = tuple(
        CandidateDraft(f"cand_{index:03d}", f"The destination is {city}.")
        for index, city in enumerate(("Oslo", "Bergen", "Paris", "Rome"), start=1)
    )
    outputs = _outputs(source, candidates)
    outputs[("memory_extraction_evidence_support_card", "cand_004")] = "none"
    calls = []

    async def select(_client, **kwargs):
        calls.append(kwargs)
        reference = catalog.resolve("r1", "r1")
        return {candidate.candidate_id: reference for candidate in kwargs["candidates"]}

    monkeypatch.setattr(
        "atagia.memory.source_quote_selector.select_source_references", select
    )
    client = _CardClient(outputs)
    card = await _run(
        client,
        source,
        candidates,
        evidence_model="typesafe/jev-latest",
        catalog=catalog,
    )

    assert len(calls) == 1
    assert calls[0]["candidates"] == candidates[:3]
    assert set(calls[0]["support_kinds"]) == {"cand_001", "cand_002", "cand_003"}
    assert card.parsed["cand_004"] == {"start_ref": None, "end_ref": None}
    assert len(client.requests) == 10


@pytest.mark.asyncio
async def test_failing_support_cancels_and_awaits_sibling() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class FailingClient:
        complete_choice_questions = LLMClient.complete_choice_questions

        async def complete(self, request):
            if request.metadata["candidate_id"] == "cand_002":
                started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            await started.wait()
            return LLMCompletionResponse(
                provider="test", model=request.model, output_text="invalid"
            )

    candidates = (
        CandidateDraft("cand_001", "One."),
        CandidateDraft("cand_002", "Two."),
    )
    with pytest.raises(LLMError, match="unknown option"):
        await asyncio.wait_for(
            _run(FailingClient(), "One. Two.", candidates), timeout=2
        )
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_failing_plain_reference_cancels_pending_support() -> None:
    source = "Oslo is the destination."
    candidates = (
        CandidateDraft("cand_001", "The destination is Oslo."),
        CandidateDraft("cand_002", "The destination is Paris."),
    )
    outputs = _outputs(source, candidates)
    outputs[("memory_extraction_source_reference_card", "cand_001")] = "r999 r100"
    support_started = asyncio.Event()
    support_cancelled = asyncio.Event()

    class FailingReferenceClient(_CardClient):
        async def complete(self, request):
            key = request.metadata["purpose"], request.metadata["candidate_id"]
            if key == ("memory_extraction_evidence_support_card", "cand_002"):
                support_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    support_cancelled.set()
                    raise
            if key == ("memory_extraction_source_reference_card", "cand_001"):
                await support_started.wait()
            return await super().complete(request)

    with pytest.raises(ValueError, match="Unknown source reference"):
        await asyncio.wait_for(
            _run(FailingReferenceClient(outputs), source, candidates), timeout=2
        )
    assert support_cancelled.is_set()


@pytest.mark.asyncio
async def test_cancelled_sibling_attempt_is_recorded(tmp_path) -> None:
    started = asyncio.Event()

    class FailingProvider(LLMProvider):
        name = "openai"

        async def complete(self, request):
            if request.metadata["candidate_id"] == "cand_002":
                started.set()
                await asyncio.Future()
            await started.wait()
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text="invalid"
            )

    recorder = DiagnosticRecorder(tmp_path)
    client = LLMClient(
        providers=[FailingProvider()],
        retry_policy=RetryPolicy(attempts=1),
        diagnostic_recorder=recorder,
        max_concurrent_requests_per_provider=2,
    )
    candidates = (
        CandidateDraft("cand_001", "One."),
        CandidateDraft("cand_002", "Two."),
    )
    try:
        with pytest.raises(LLMError, match="unknown option"):
            await asyncio.wait_for(_run(client, "One. Two.", candidates), timeout=2)
    finally:
        await client.aclose()
        recorder.close()
    events = [
        json.loads(line)
        for line in (recorder.root / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        event["kind"] == "provider_attempt"
        and event.get("phase") == "end"
        and event["status"] == "cancelled"
        for event in events
    )
    operations = [event for event in events if event["kind"] == "operation_start"]
    assert {event["purpose"] for event in operations} == {
        "memory_extraction_evidence",
        "memory_extraction_evidence_support_card",
    }


@pytest.mark.asyncio
async def test_typed_support_and_preservation_share_only_ready_candidates(tmp_path) -> None:
    source = "Lena said yes to Porto. Malik did not agree to Paris."
    candidates = (
        CandidateDraft("cand_001", "Lena agreed to Porto."),
        CandidateDraft("cand_002", "Malik agreed to Paris."),
        CandidateDraft("cand_003", "Malik did not agree to Paris."),
    )
    active = 0
    max_active = 0
    requests = []

    class DecisionProvider(LLMProvider):
        name = "typesafe"
        supports_choices = True

        async def complete(self, request):
            nonlocal active, max_active
            requests.append(request)
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.005)
            finally:
                active -= 1
            purpose = request.metadata["purpose"]
            values = (
                {
                    "cand_001": "contextual_direct",
                    "cand_002": "none",
                    "cand_003": "direct",
                }
                if purpose == "memory_extraction_evidence_support_card"
                else {"cand_001": "yes", "cand_003": "no"}
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                choice_answers={
                    candidate_id: ChoiceAnswer(
                        type="choice",
                        choice=values[candidate_id],
                        probabilities={values[candidate_id]: 1.0},
                        confidence=1.0,
                    )
                    for candidate_id in request.choice_questions
                },
            )

    class LanguageAndReferenceProvider(LLMProvider):
        name = "openai"

        async def complete(self, request):
            nonlocal active, max_active
            requests.append(request)
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.005)
            finally:
                active -= 1
            purpose = request.metadata["purpose"]
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=(
                    "en\nes"
                    if purpose == "memory_extraction_candidate_language_card"
                    else "r1 r4"
                ),
            )

    recorder = DiagnosticRecorder(tmp_path)
    client = LLMClient(
        providers=[DecisionProvider(), LanguageAndReferenceProvider()],
        retry_policy=RetryPolicy(attempts=1),
        diagnostic_recorder=recorder,
    )
    try:
        result = await _run(
            client,
            source,
            candidates,
            support_model="typesafe/jev-1.13.0",
            preserve_model="typesafe/jev-1.13.0",
            semaphore=asyncio.Semaphore(1),
        )
    finally:
        await client.aclose()
        recorder.close()

    assert max_active == 1
    support = [
        request
        for request in requests
        if request.metadata["purpose"] == "memory_extraction_evidence_support_card"
    ]
    preservation = [
        request
        for request in requests
        if request.metadata["purpose"] == "memory_extraction_preserve_verbatim_card"
    ]
    assert len(support) == len(preservation) == 1
    assert set(support[0].choice_questions) == {"cand_001", "cand_002", "cand_003"}
    assert set(preservation[0].choice_questions) == {"cand_001", "cand_003"}
    assert all(
        candidate.canonical_text
        in support[0].choice_questions[candidate.candidate_id].instructions
        for candidate in candidates
    )
    assert source in support[0].messages[-1].content
    assert support[0].metadata["candidate_ids"] == tuple(
        candidate.candidate_id for candidate in candidates
    )
    assert result.parsed["cand_002"] == {"start_ref": None, "end_ref": None}
    assert result.parsed["cand_001"]["support_kind"] == "contextual_direct"
    assert result.parsed["cand_001"]["preserve_verbatim"] is True
    assert result.parsed["cand_003"]["preserve_verbatim"] is False
    assert result.parsed["cand_001"]["language_codes"] == ("en", "es")
    assert not any(
        request.metadata.get("candidate_id") == "cand_002"
        and request.metadata["purpose"] != "memory_extraction_evidence_support_card"
        for request in requests
    )
    operations = [
        json.loads(line)
        for line in (recorder.root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {
        event["purpose"]: event["component"]
        for event in operations
        if event["kind"] == "operation_start" and event["purpose"] in {
            "memory_extraction_evidence_support_card",
            "memory_extraction_preserve_verbatim_card",
        }
    } == {
        "memory_extraction_evidence_support_card": "extraction_evidence_support",
        "memory_extraction_preserve_verbatim_card": "extraction_preserve_verbatim",
    }


@pytest.mark.asyncio
async def test_typed_evidence_path_reuses_decided_support_and_source_catalog() -> None:
    source = "Mira did not approve the transfer unless the amount is below $50."
    catalog = SourceReferenceCatalog(source)
    candidates = (
        CandidateDraft(
            "cand_001", "Mira did not approve the transfer unconditionally."
        ),
        CandidateDraft("cand_002", "Mira approved every transfer."),
    )
    requests = []

    class DecisionProvider(LLMProvider):
        name = "typesafe"
        supports_choices = True

        async def complete(self, request):
            requests.append(request)
            purpose = request.metadata["purpose"]
            if purpose == "memory_extraction_evidence_support_card":
                values = {"cand_001": "direct", "cand_002": "none"}
            elif purpose == "memory_extraction_preserve_verbatim_card":
                values = {"cand_001": "yes"}
            else:
                values = {
                    "cand_001.start": catalog.anchors[0].reference_id,
                    "cand_001.end": catalog.anchors[-1].reference_id,
                }
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                choice_answers={
                    question_id: ChoiceAnswer(
                        type="choice",
                        choice=values[question_id],
                        probabilities={values[question_id]: 1.0},
                        confidence=1.0,
                    )
                    for question_id in request.choice_questions
                },
            )

    class LanguageProvider(LLMProvider):
        name = "openai"

        async def complete(self, request):
            requests.append(request)
            assert (
                request.metadata["purpose"]
                == "memory_extraction_candidate_language_card"
            )
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text="en"
            )

    client = LLMClient(
        providers=[DecisionProvider(), LanguageProvider()],
        retry_policy=RetryPolicy(attempts=1),
    )
    try:
        result = await _run(
            client,
            source,
            candidates,
            support_model="typesafe/jev-1.13.0",
            preserve_model="typesafe/jev-1.13.0",
            evidence_model="typesafe/jev-1.13.0",
            catalog=catalog,
            semaphore=asyncio.Semaphore(1),
        )
    finally:
        await client.aclose()

    purposes = [request.metadata["purpose"] for request in requests]
    assert purposes.count("memory_extraction_evidence_support_card") == 1
    assert purposes.count("memory_extraction_preserve_verbatim_card") == 1
    assert purposes.count("memory_extraction_source_reference_selector") == 1
    assert purposes.count("memory_extraction_candidate_language_card") == 1
    assert "memory_extraction_source_reference_card" not in purposes
    reference_request = next(
        request
        for request in requests
        if request.metadata["purpose"] == "memory_extraction_source_reference_selector"
    )
    assert (
        "Earlier discussion concerned travel plans."
        in reference_request.messages[-1].content
    )
    assert catalog.render() in reference_request.messages[-1].content
    assert result.parsed["cand_002"] == {"start_ref": None, "end_ref": None}
    row = result.parsed["cand_001"]
    assert row["support_kind"] == "direct" and row["preserve_verbatim"] is True
    assert catalog.resolve(row["start_ref"], row["end_ref"]).quote(source) == source
