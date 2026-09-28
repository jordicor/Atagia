"""Offline contract tests for compact source-reference challengers."""

from __future__ import annotations

from typing import Any

import pytest

from atagia.core.source_references import SourceReferenceCatalog, source_sha256
from atagia.memory.extraction_cards import CandidateDraft
from atagia.memory.source_quote_selector import _TASK
from atagia.models.schemas_decisions import ChoiceAnswer
from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from benchmarks.source_quote_selection.compact_selector import select_source_references


class CannedClient:
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
                    type="choice", choice=value, probabilities={value: 1.0}, confidence=1.0
                )
                for key, value in reply.items()
            },
        )


def _candidate(candidate_id: str = "cand_001") -> CandidateDraft:
    return CandidateDraft(candidate_id=candidate_id, canonical_text="Proposed fact")


def _context(source: str) -> str:
    return (
        '<source_message role="user">\n'
        f"<message_text>\n{SourceReferenceCatalog(source).render()}\n</message_text>\n"
        "</source_message>\n"
        "<recent_context>Earlier turn remains available.</recent_context>\n"
        "<prior_chunk_context>Earlier chunk remains available.</prior_chunk_context>"
    )


async def _select(
    client: CannedClient,
    source: str,
    *,
    mode: str,
    candidates: tuple[CandidateDraft, ...] = (_candidate(),),
    context: str | None = None,
) -> dict[str, Any]:
    return await select_source_references(  # type: ignore[arg-type]
        client,
        model="typesafe/jev-latest",
        source_text=source,
        candidates=candidates,
        support_kinds={candidate.candidate_id: "direct" for candidate in candidates},
        source_context=context if context is not None else _context(source),
        mode=mode,
    )


@pytest.mark.parametrize("mode", ["compact", "focused"])
@pytest.mark.parametrize("count", [12, 254])
async def test_small_sources_use_one_call_and_null_reference_descriptions(
    mode: str, count: int,
) -> None:
    source = " ".join(f"item_{index:03d}" for index in range(count))
    client = CannedClient([{"cand_001.start": "r2", "cand_001.end": "r3"}])

    result = await _select(client, source, mode=mode)

    assert len(client.requests) == 1
    request = client.requests[0]
    assert request.messages[0].content == _TASK
    assert request.messages[1].content.startswith(_context(source))
    assert request.metadata["purpose"] == "memory_extraction_source_reference_selector"
    assert request.metadata["challenger_mode"] == mode
    assert result["cand_001"].quote(source) == "item_001 item_002"
    assert all(
        len(question.criteria) == count + 1
        and question.criteria["none"] == "The source message does not support the candidate"
        and all(value is None for key, value in question.criteria.items() if key != "none")
        for question in request.choice_questions.values()
    )


@pytest.mark.parametrize("mode", ["compact", "focused"])
async def test_255_anchors_use_two_calls_with_original_ids(mode: str) -> None:
    source = " ".join(f"item_{index:03d}" for index in range(255))
    end_block = "b2" if mode == "compact" else "b4"
    client = CannedClient(
        [
            {"cand_001.start": "b1", "cand_001.end": end_block},
            {"cand_001.start": "r1", "cand_001.end": "r255"},
        ]
    )

    result = await _select(client, source, mode=mode)

    assert len(client.requests) == 2
    assert result["cand_001"].quote(source) == source
    assert all(
        len(question.criteria) <= 255
        for request in client.requests
        for question in request.choice_questions.values()
    )
    assert all(
        value is None
        for question in client.requests[1].choice_questions.values()
        for key, value in question.criteria.items()
        if key != "none"
    )


async def test_focused_refinement_keeps_all_intermediate_blocks_and_whitespace() -> None:
    source = "\r\n".join(f"item_{index:03d}" for index in range(320))
    client = CannedClient(
        [
            {"cand_001.start": "b1", "cand_001.end": "b3"},
            {"cand_001.start": "r63", "cand_001.end": "r130"},
        ]
    )

    result = await _select(client, source, mode="focused")

    context = client.requests[1].messages[1].content
    assert '[r100]item_099' in context
    assert '[r193]item_192' not in context
    assert '<included_source_region first="b1" last="b3">' in context
    assert '<omitted_source_blocks first="b4" last="b5"/>' in context
    assert "\r\n" in context
    assert "Earlier turn remains available" in context
    assert "Earlier chunk remains available" in context
    assert result["cand_001"].quote(source) == source[
        source.index("item_062") : source.index("item_129") + len("item_129")
    ]


async def test_focused_refinement_marks_gaps_between_candidates() -> None:
    source = " ".join(f"item_{index:03d}" for index in range(320))
    candidates = (_candidate("cand_001"), _candidate("cand_002"))
    client = CannedClient(
        [
            {
                "cand_001.start": "b1", "cand_001.end": "b1",
                "cand_002.start": "b4", "cand_002.end": "b4",
            },
            {
                "cand_001.start": "r1", "cand_001.end": "r2",
                "cand_002.start": "r193", "cand_002.end": "r194",
            },
        ]
    )

    result = await _select(client, source, mode="focused", candidates=candidates)

    context = client.requests[1].messages[1].content
    assert '<omitted_source_blocks first="b2" last="b3"/>' in context
    assert "[r65]item_064" not in context
    assert "[r193]item_192" in context
    assert '<candidate id="cand_001" support="direct">' in context
    assert '<candidate id="cand_002" support="direct">' in context
    assert result["cand_001"].quote(source) == "item_000 item_001"
    assert result["cand_002"].quote(source) == "item_192 item_193"


@pytest.mark.parametrize("mode", ["compact", "focused"])
async def test_batch_keeps_none_candidate_out_of_refinement(mode: str) -> None:
    source = " ".join(f"item_{index:03d}" for index in range(255))
    candidates = (_candidate("cand_001"), _candidate("cand_002"))
    client = CannedClient(
        [
            {
                "cand_001.start": "b1", "cand_001.end": "b1",
                "cand_002.start": "none", "cand_002.end": "none",
            },
            {"cand_001.start": "r1", "cand_001.end": "r2"},
        ]
    )

    result = await _select(client, source, mode=mode, candidates=candidates)

    assert result["cand_001"].quote(source) == "item_000 item_001"
    assert result["cand_002"] is None
    assert set(client.requests[1].choice_questions) == {"cand_001.start", "cand_001.end"}


async def test_unicode_crlf_and_apparent_ids_resolve_only_catalog_anchors() -> None:
    source = "Élodie wrote café.\r\n[r999] Élodie wrote café."
    catalog = SourceReferenceCatalog(source)
    second = source.index("Élodie", 1)
    first_ref = next(a.reference_id for a in catalog.anchors if a.char_start == second)
    last_ref = catalog.anchors[-1].reference_id
    client = CannedClient([{"cand_001.start": first_ref, "cand_001.end": last_ref}])

    result = await _select(client, source, mode="compact")

    reference = result["cand_001"]
    assert reference.quote(source) == "Élodie wrote café."
    assert reference.source_sha256 == source_sha256(source)
    assert "&#91;" in client.requests[0].messages[1].content
    assert "\r\n" in client.requests[0].messages[1].content
    assert "r999" not in client.requests[0].choice_questions["cand_001.start"].criteria


@pytest.mark.parametrize("mode", ["compact", "focused"])
@pytest.mark.parametrize(
    "replies",
    [
        [{"cand_001.start": "none", "cand_001.end": "r1"}],
        [{"cand_001.start": "r2", "cand_001.end": "r1"}],
        [{"cand_001.start": "r999", "cand_001.end": "r1"}],
        [{"cand_001.start": "r1"}],
    ],
)
async def test_invalid_boundary_answers_fail(mode: str, replies: list[dict[str, str]]) -> None:
    with pytest.raises(ValueError):
        await _select(CannedClient(replies), "alpha beta", mode=mode)


@pytest.mark.parametrize("mode", ["compact", "focused"])
async def test_reversed_blocks_fail_before_refinement(mode: str) -> None:
    source = " ".join(f"item_{index:03d}" for index in range(255))
    last = "b2" if mode == "compact" else "b4"
    client = CannedClient([{"cand_001.start": last, "cand_001.end": "b1"}])
    with pytest.raises(ValueError, match="reversed blocks"):
        await _select(client, source, mode=mode)
    assert len(client.requests) == 1


async def test_focused_block_capacity_fails_before_dispatch() -> None:
    source = " ".join("x" for _ in range(254 * 64 + 1))
    client = CannedClient([])
    with pytest.raises(ValueError, match="block capacity"):
        await _select(client, source, mode="focused")
    assert client.requests == []


async def test_snapshot_mismatch_fails_before_dispatch() -> None:
    client = CannedClient([])
    with pytest.raises(ValueError, match="exactly the referenced source"):
        await _select(client, "current source", mode="focused", context=_context("other source"))
    assert client.requests == []
