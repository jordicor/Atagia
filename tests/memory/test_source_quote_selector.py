"""Finite-choice contract checks for literal source-reference selection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from atagia.core.source_references import SourceReferenceCatalog, source_sha256
from atagia.memory.extraction_cards import CandidateDraft
from atagia.memory.source_quote_selector import select_source_references
from atagia.models.schemas_decisions import ChoiceAnswer
from atagia.services.llm_client import (
    ConfigurationError,
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMError,
    LLMProvider,
    RetryPolicy,
)


class FiniteClient:
    complete_choice_questions = LLMClient.complete_choice_questions

    def __init__(self, replies: list[dict[str, str]]) -> None:
        self.replies = replies
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        reply = self.replies[len(self.requests) - 1]
        return LLMCompletionResponse(
            provider="typesafe",
            model=request.model,
            choice_answers={
                key: ChoiceAnswer(
                    type="choice",
                    choice=value,
                    probabilities={value: 1.0},
                    confidence=1.0,
                )
                for key, value in reply.items()
            },
        )


def _candidate(candidate_id: str = "cand_001") -> CandidateDraft:
    return CandidateDraft(
        candidate_id=candidate_id, canonical_text="A proposed source fact."
    )


def _context(source: str, recent: str = "Earlier conversation is visible.") -> str:
    rendered = SourceReferenceCatalog(source).render()
    return (
        '<source_message role="user">\n'
        f"<message_text>\n{rendered}\n</message_text>\n"
        "</source_message>\n"
        f"<recent_context>{recent}</recent_context>\n"
        "<prior_chunk_context>Prior facts remain visible.</prior_chunk_context>"
    )


def _refs_for(source: str, phrase: str, occurrence: int = 0) -> tuple[str, str]:
    start = -1
    for _ in range(occurrence + 1):
        start = source.find(phrase, start + 1)
        assert start >= 0
    end = start + len(phrase)
    catalog = SourceReferenceCatalog(source)
    first = next(
        anchor.reference_id for anchor in catalog.anchors if anchor.char_start == start
    )
    last = next(
        anchor.reference_id for anchor in catalog.anchors if anchor.char_end == end
    )
    return first, last


async def _select(
    client: FiniteClient,
    source: str,
    *,
    context: str | None = None,
    candidates: tuple[CandidateDraft, ...] | None = None,
    support_kinds: dict[str, str] | None = None,
):
    selected_candidates = candidates or (_candidate(),)
    return await select_source_references(
        client,
        model="typesafe/jev-latest",
        source_text=source,
        candidates=selected_candidates,
        support_kinds=(
            support_kinds
            if support_kinds is not None
            else {candidate.candidate_id: "direct" for candidate in selected_candidates}
        ),
        source_context=context if context is not None else _context(source),
    )


@pytest.mark.asyncio
async def test_repeated_unicode_source_keeps_exact_crlf_and_snapshot() -> None:
    source = "Élodie wrote café.\r\nÉlodie wrote café."
    first, last = _refs_for(source, "Élodie wrote café.", occurrence=1)
    client = FiniteClient([{"cand_001.start": first, "cand_001.end": last}])

    result = await _select(client, source)

    reference = result["cand_001"]
    assert reference is not None
    assert reference.quote(source) == "Élodie wrote café."
    assert reference.char_start == source.index("Élodie", 1)
    assert reference.source_sha256 == source_sha256(source)
    assert "\r\n" in client.requests[0].messages[1].content


@pytest.mark.asyncio
async def test_supported_candidates_share_one_native_reference_request() -> None:
    source = "Oslo and Bergen."
    candidates = (_candidate("cand_001"), _candidate("cand_002"))
    client = FiniteClient(
        [
            {
                "cand_001.start": "r1",
                "cand_001.end": "r1",
                "cand_002.start": "r3",
                "cand_002.end": "r3",
            }
        ]
    )

    result = await _select(client, source, candidates=candidates)

    assert len(client.requests) == 1
    assert set(client.requests[0].choice_questions) == {
        "cand_001.start",
        "cand_001.end",
        "cand_002.start",
        "cand_002.end",
    }
    assert result["cand_001"].quote(source) == "Oslo"
    assert result["cand_002"].quote(source) == "Bergen"


@pytest.mark.asyncio
async def test_selected_passage_preserves_crlf_inside_the_quote() -> None:
    source = "Élodie wrote café.\r\nÉlodie wrote café."
    passage = "café.\r\nÉlodie"
    first, last = _refs_for(source, passage)
    client = FiniteClient([{"cand_001.start": first, "cand_001.end": last}])

    result = await _select(client, source)

    assert result["cand_001"] is not None
    assert result["cand_001"].quote(source) == passage


@pytest.mark.asyncio
async def test_recent_context_visible_but_only_current_message_ids_are_choices() -> (
    None
):
    source = "The east one."
    recent = "Assistant asked which gate to use; archived text had [r999] and [r1]."
    first, last = _refs_for(source, "east one")
    client = FiniteClient([{"cand_001.start": first, "cand_001.end": last}])

    result = await _select(
        client,
        source,
        context=_context(source, recent),
        support_kinds={"cand_001": "contextual_direct"},
    )

    assert result["cand_001"] is not None
    request = client.requests[0]
    assert recent in request.messages[1].content
    assert "Prior facts remain visible" in request.messages[1].content
    assert (
        "contextual_direct" in request.choice_questions["cand_001.start"].instructions
    )
    assert (
        _candidate().canonical_text
        in request.choice_questions["cand_001.start"].instructions
    )
    current_refs = {
        anchor.reference_id for anchor in SourceReferenceCatalog(source).anchors
    }
    for question in request.choice_questions.values():
        assert set(question.criteria) == current_refs | {"none"}
        assert "r999" not in question.criteria


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "support_kinds",
    [{}, {"other_candidate": "direct"}],
    ids=["assessment-absent", "wrong-candidate-id"],
)
async def test_missing_or_wrong_assessment_rejected_before_dispatch(
    support_kinds: dict[str, str],
) -> None:
    source = "The east one."
    client = FiniteClient([])

    with pytest.raises(ValueError, match="each candidate's support assessment"):
        await _select(client, source, support_kinds=support_kinds)

    assert client.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context_factory",
    [
        lambda source: _context("A different source."),
        lambda source: _context(source) + "\n" + _context(source),
        lambda source: _context(source).replace("[r1]", "[r999]", 1),
    ],
    ids=["wrong-snapshot", "duplicate-snapshot", "altered-anchor"],
)
async def test_incorrect_source_snapshot_is_rejected_before_dispatch(
    context_factory: Callable[[str], str],
) -> None:
    source = "The current source."
    client = FiniteClient([])

    with pytest.raises(ValueError, match="exactly the referenced source"):
        await _select(client, source, context=context_factory(source))

    assert client.requests == []


@pytest.mark.asyncio
async def test_no_support_requires_none_at_both_ends() -> None:
    client = FiniteClient([{"cand_001.start": "none", "cand_001.end": "none"}])

    result = await _select(client, "The note records a weekday meeting.")

    assert result == {"cand_001": None}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answers", "message"),
    [
        ({"cand_001.start": "none", "cand_001.end": "r2"}, "inconsistent absence"),
        ({"cand_001.start": "r1", "cand_001.end": "none"}, "inconsistent absence"),
        ({"cand_001.start": "r1"}, "answer IDs"),
        (
            {"cand_001.start": "r1", "cand_001.end": "r2", "extra.start": "r1"},
            "answer IDs",
        ),
        ({"cand_001.start": "r999", "cand_001.end": "r2"}, "unknown option"),
        ({"cand_001.start": "r2", "cand_001.end": "r1"}, "source order"),
    ],
)
async def test_invalid_small_source_answer_fails_fast(
    answers: dict[str, str], message: str
) -> None:
    client = FiniteClient([answers])

    with pytest.raises((ValueError, LLMError), match=message):
        await _select(client, "alpha beta")


@pytest.mark.asyncio
async def test_large_source_uses_two_stages_and_keeps_context_across_blocks() -> None:
    source = " ".join(f"term_{index:03d}" for index in range(280))
    assert len(SourceReferenceCatalog(source).anchors) == 280
    client = FiniteClient(
        [
            {"cand_001.start": "b1", "cand_001.end": "b2"},
            {"cand_001.start": "r253", "cand_001.end": "r257"},
        ]
    )
    recent = "Assistant asked about the terms near the page boundary."

    result = await _select(client, source, context=_context(source, recent))

    reference = result["cand_001"]
    assert reference is not None
    assert reference.quote(source) == " ".join(
        f"term_{index:03d}" for index in range(252, 257)
    )
    assert [request.metadata["stage"] for request in client.requests] == [
        "blocks",
        "anchors",
    ]
    for request in client.requests:
        assert recent in request.messages[1].content
        assert "Prior facts remain visible" in request.messages[1].content
        assert '<block id="b1"' in request.messages[1].content
        assert '<block id="b2"' in request.messages[1].content
    first, second = client.requests
    assert set(first.choice_questions["cand_001.start"].criteria) == {
        "none",
        "b1",
        "b2",
    }
    assert "r253" in second.choice_questions["cand_001.start"].criteria
    assert "r257" in second.choice_questions["cand_001.end"].criteria
    assert "r257" not in second.choice_questions["cand_001.start"].criteria
    assert "r253" not in second.choice_questions["cand_001.end"].criteria


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_answers", "second_answers", "message"),
    [
        (
            {"cand_001.start": "b1", "cand_001.end": "b2"},
            {"cand_001.start": "none", "cand_001.end": "none"},
            "contradict the selected support blocks",
        ),
        (
            {"cand_001.start": "b1", "cand_001.end": "none"},
            None,
            "inconsistent absence",
        ),
        (
            {"cand_001.start": "b2", "cand_001.end": "b1"},
            None,
            "reversed blocks",
        ),
    ],
)
async def test_large_source_rejects_block_boundary_contradictions(
    first_answers: dict[str, str],
    second_answers: dict[str, str] | None,
    message: str,
) -> None:
    source = " ".join(f"item_{index:03d}" for index in range(270))
    client = FiniteClient(
        [first_answers] + ([second_answers] if second_answers else [])
    )

    with pytest.raises(ValueError, match=message):
        await _select(client, source)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate_order", [("cand_z", "cand_a"), ("cand_a", "cand_z")]
)
async def test_long_conditional_contract_uses_full_source_and_literal_cross_block_quote(
    candidate_order: tuple[str, str],
) -> None:
    prefix = " ".join(f"term_{index:03d}" for index in range(252))
    suffix = " ".join(f"term_{index:03d}" for index in range(262, 290))
    passage = "unless Mira says yes, do not transfer funds"
    source = f"{prefix} {passage}. {suffix}"
    first, last = _refs_for(source, passage)
    catalog = SourceReferenceCatalog(source)
    assert len(catalog.anchors) > 254
    requests = []

    class ReferenceProvider(LLMProvider):
        name = "typesafe"
        supports_choices = True

        async def complete(self, request):
            requests.append(request)
            stage = request.metadata["stage"]
            values = (
                {
                    "cand_z.start": "b1",
                    "cand_z.end": "b2",
                    "cand_a.start": "none",
                    "cand_a.end": "none",
                }
                if stage == "blocks"
                else {"cand_z.start": first, "cand_z.end": last}
            )
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

    client = LLMClient(
        providers=[ReferenceProvider()], retry_policy=RetryPolicy(attempts=1)
    )
    by_id = {
        "cand_z": CandidateDraft("cand_z", "Funds must not move unless Mira approves."),
        "cand_a": CandidateDraft(
            "cand_a", "The contract says all transfers are unconditional."
        ),
    }
    candidates = tuple(by_id[candidate_id] for candidate_id in candidate_order)
    try:
        selected = await asyncio.wait_for(
            select_source_references(
                client,
                model="typesafe/jev-1.13.0",
                source_text=source,
                source_catalog=catalog,
                candidates=candidates,
                support_kinds={"cand_z": "inferred", "cand_a": "weak_signal"},
                source_context=_context(source, "Mira is the only approver."),
                metadata={"user_id": "usr_1"},
                dispatch_semaphore=asyncio.Semaphore(1),
            ),
            timeout=2,
        )
    finally:
        await client.aclose()

    assert [request.metadata["stage"] for request in requests] == ["blocks", "anchors"]
    assert all(request.metadata["user_id"] == "usr_1" for request in requests)
    assert all(
        "term_000" in request.messages[1].content
        and "term_289" in request.messages[1].content
        for request in requests
    )
    assert all(
        "Mira is the only approver." in request.messages[1].content
        for request in requests
    )
    assert (
        by_id["cand_z"].canonical_text
        in requests[0].choice_questions["cand_z.start"].instructions
    )
    assert "inferred" in requests[0].choice_questions["cand_z.start"].instructions
    assert set(requests[1].choice_questions) == {"cand_z.start", "cand_z.end"}
    assert selected["cand_a"] is None
    assert selected["cand_z"] is not None
    assert selected["cand_z"].quote(source) == passage
    assert selected["cand_z"].source_sha256 == source_sha256(source)


@pytest.mark.asyncio
async def test_source_over_native_context_bound_fails_without_truncating() -> None:
    source = " ".join(f"token_{index:04d}" for index in range(4000))
    client = FiniteClient([])

    with pytest.raises(ConfigurationError, match="safe 32k-token bound"):
        await _select(client, source)

    assert client.requests == []
