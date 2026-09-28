from pathlib import Path
import asyncio

import pytest

from benchmarks.memory_extraction_cards.compare import (
    BenchmarkCase,
    ExpectedCandidate,
    load_cases,
    normalize_result,
    run_cards_variant,
    run_one_variant,
    score_output,
)
from atagia.memory.extraction_cards import (
    CandidateDraft, CardResult, _CARD_SYSTEM_PROMPTS, assemble_card_result,
    parse_candidate_card_output, parse_classification_output, parse_coverage_members_card_output,
)
from atagia.services.llm_client import LLMCompletionResponse
from atagia.memory.coverage_members_card import MEMBERS_SYSTEM_PROMPT
from atagia.memory.extraction_temporal import (
    parse_temporal_interval_output,
    parse_temporal_type_output,
)
from atagia.core.source_references import SourceReferenceCatalog
from tests.extraction_payload_support import memory_extraction_card_output_from_payload


_LINE_ONLY_CARD_SYSTEM_PROMPT = (
    "Extract durable memory as plain-text card lines. "
    "Write only the requested lines. No JSON. No explanation."
)
def test_memory_extraction_card_system_prompts_are_card_specific() -> None:
    assert _CARD_SYSTEM_PROMPTS["coverage_members"] == MEMBERS_SYSTEM_PROMPT
    for card_name in (
        "candidate",
        "index",
    ):
        assert _CARD_SYSTEM_PROMPTS[card_name] == _LINE_ONLY_CARD_SYSTEM_PROMPT
    assert "two JSON lines" in _CARD_SYSTEM_PROMPTS["temporal_interval"]
    assert "canonical claim key" in _CARD_SYSTEM_PROMPTS["belief_key"]
    assert "literal claim value" in _CARD_SYSTEM_PROMPTS["belief_value"]
    for card_name in ("memory_kind", "memory_scope", "memory_confidence"):
        assert "one property" in _CARD_SYSTEM_PROMPTS[card_name]


def test_memory_extraction_cards_case_set_loads() -> None:
    cases = load_cases(Path("benchmarks/memory_extraction_cards/cases.jsonl"))

    assert len(cases) == 100
    assert {case.case_id for case in cases} >= {
        "none_greeting",
        "multiple_atomic_facts",
        "relative_event_yesterday",
        "belief_claim",
        "translation_request_no_memory",
        "future_event_tomorrow",
        "preference_changed_now",
        "french_cafe_preference",
        "chat_only_staging_placeholder",
        "contextual_workshop_name",
        "current_hotel_until_sunday",
        "temporary_phone_until_tuesday",
        "assistant_found_query_cache_cause",
        "office_wifi_password_placeholder",
        "api_language_contract",
        "ship_it_means_local_commit",
        "one_time_weather_no_memory",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("variant,cap", [("cards_serial", 1), ("cards_bounded_2", 2), ("cards_parallel", 8)])
async def test_shadow_runner_executes_the_complete_production_graph(variant, cap) -> None:
    source = "I want choices explained with trade-offs. Yesterday I met Dr. Jo."
    payload = {"candidates": [
        {
            "canonical_text": "The user wants choices explained with trade-offs.",
            "kind": "belief", "subject_scope": "user", "confidence": 0.88,
            "source_span": "I want choices explained with trade-offs.",
            "language_codes": ["en"], "claim_key": "decisions.tradeoffs",
            "claim_value": "explain trade-offs", "support_kind": "direct",
        },
        {
            "canonical_text": "The user met Dr. Jo yesterday.",
            "kind": "evidence", "subject_scope": "user", "confidence": 0.91,
            "source_span": "Yesterday I met Dr. Jo.", "language_codes": ["en"],
            "temporal_status": {"type": "event_triggered", "valid_from_iso": "2026-09-25T00:00:00-04:00", "valid_to_iso": "2026-09-25T23:59:59-04:00"},
            "coverage_members": [{"member_key": "dr. jo", "display_text": "Dr. Jo"}],
        },
    ]}

    class Client:
        def __init__(self):
            self.requests = []
            self.active = self.peak = 0

        async def complete(self, request):
            self.requests.append(request)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0)
                return LLMCompletionResponse(
                    provider="openrouter", model=request.model,
                    output_text=memory_extraction_card_output_from_payload(
                        payload, request.metadata["purpose"],
                        prompt="\n".join(message.content for message in request.messages),
                    ),
                )
            finally:
                self.active -= 1

    client = Client()
    result, calls, repairs = await run_cards_variant(
        client=client,
        case=BenchmarkCase(case_id="complete_graph", message=source, expected_candidates=(), occurred_at="2026-09-26T12:00:00-04:00"),
        model="openrouter/openai/gpt-6-luna", variant=variant, include_examples=False,
    )
    assert repairs == []
    assert client.peak <= cap and client.active == 0
    assert len(calls) == len(client.requests)
    assert all(call["response"] is not None and call["error"] is None for call in calls)
    purposes = [call["purpose"] for call in calls]
    assert purposes.count("memory_extraction_belief_key_card") == 1
    assert purposes.count("memory_extraction_belief_value_card") == 1
    assert purposes.count("memory_extraction_temporal_interval_card") == 1
    assert purposes.count("memory_extraction_coverage_member_identity_card") == 1
    assert purposes.count("memory_extraction_source_reference_card") == 2
    assert result.candidates[0].claim_value == "explain trade-offs"
    assert result.candidates[0].source_reference.quote(source) == "I want choices explained with trade-offs."
    assert result.candidates[1].coverage_members[0].member_key == "dr. jo"
    assert result.candidates[1].temporal_status.valid_to_iso == "2026-09-25T23:59:59-04:00"


@pytest.mark.asyncio
async def test_shadow_runner_preserves_calls_when_production_validation_fails() -> None:
    class Client:
        async def complete(self, request):
            purpose = request.metadata["purpose"]
            output = "cand_001 | The user likes tea." if purpose == "memory_extraction_candidate_card" else "invalid"
            return LLMCompletionResponse(provider="openrouter", model=request.model, output_text=output)

    row = await run_one_variant(
        client=Client(), case=BenchmarkCase("invalid_reply", "I like tea.", ()),
        variant="cards_serial", card_model="openrouter/openai/gpt-6-luna",
        repetition=1, include_examples=False,
    )
    assert row["technical_ok"] is False
    assert row["error"] is not None
    assert row["card_calls"][0]["response"]["output_text"].startswith("cand_001")
    assert any(call["response"] and call["response"]["output_text"] == "invalid" for call in row["card_calls"])


def test_candidate_card_parser_accepts_plain_lines() -> None:
    candidates, malformed = parse_candidate_card_output(
        "cand_001 | User lives in Valencia.\n"
        "cand_002 | User's locker code is 7426."
    )

    assert malformed == 0
    assert [candidate.candidate_id for candidate in candidates] == ["cand_001", "cand_002"]
    assert candidates[1].canonical_text == "User's locker code is 7426."


def test_candidate_card_parser_none() -> None:
    candidates, malformed = parse_candidate_card_output("none")

    assert candidates == ()
    assert malformed == 0


def test_enrichment_parsers_accept_single_answers() -> None:
    assert parse_classification_output("memory_kind", "evidence") == "evidence"
    assert parse_classification_output("memory_scope", "user") == "user"
    assert parse_temporal_type_output("event_triggered") == "event_triggered"
    start, end = parse_temporal_interval_output('{"text":null,"time":"00:00:00","offset":"+00:00","period":"day"}\nnull')
    assert start.time == "00:00:00" and start.offset == "+00:00" and end is None
    members, malformed = parse_coverage_members_card_output(
        '"Dr. Navarro; cardiology, clinic A | room 3"\n"none"'
    )
    assert malformed == 0
    assert members == ["Dr. Navarro; cardiology, clinic A | room 3", "none"]


def test_assemble_cards_to_lean_result_and_normalized_output() -> None:
    source_text = "User's rollback phrase is RIVER-19-BLUE."
    catalog = SourceReferenceCatalog(source_text)
    code_anchors = [
        anchor for anchor in catalog.anchors
        if anchor.char_start >= source_text.index("RIVER")
        and anchor.char_end <= source_text.index("BLUE") + len("BLUE")
    ]
    candidates = (
        CandidateDraft("cand_001", "User's rollback phrase is RIVER-19-BLUE."),
    )
    result, repairs = assemble_card_result(
        candidates,
        [
            CardResult("memory_kind", "evidence", {"cand_001": "evidence"}),
            CardResult("memory_scope", "user", {"cand_001": "user"}),
            CardResult("memory_confidence", "0.9", {"cand_001": 0.9}),
            CardResult(
                "evidence",
                "",
                {
                    "cand_001": {
                        "support_kind": "direct",
                        "preserve_verbatim": True,
                        "language_codes": ("en",),
                        "start_ref": code_anchors[0].reference_id,
                        "end_ref": code_anchors[-1].reference_id,
                    }
                },
            ),
            CardResult("index", "", {"cand_001": "User's emergency rollback phrase"}),
            CardResult("temporal", "", {"cand_001": None}),
            CardResult("belief", "", {}),
            CardResult("coverage_members", "", {"cand_001": []}),
        ],
        source_catalog=catalog,
    )

    assert repairs == []
    output = normalize_result(result)
    row = output["candidates"][0]
    assert row["preserve_verbatim"] is True
    assert row["source_span"] == "RIVER-19-BLUE"
    assert row["index_text"] == "User's emergency rollback phrase"


def test_score_output_matches_expected_candidate() -> None:
    output = {
        "nothing_durable": False,
        "candidate_count": 1,
        "candidates": [
            {
                "kind": "evidence",
                "subject_scope": "user",
                "canonical_text": "User's current city is Valencia.",
                "source_span": "current city is Valencia",
                "support_kind": "direct",
                "preserve_verbatim": False,
                "language_codes": ["en"],
                "temporal_type": "unknown",
                "valid_from_iso": None,
            }
        ],
    }
    case = load_cases(Path("benchmarks/memory_extraction_cards/cases.jsonl"))[0]
    expected_case = case.__class__(
        case_id="city",
        message="My current city is Valencia.",
        expected_candidates=(
            ExpectedCandidate(
                label="city",
                kind="evidence",
                scope="user",
                must_include=("valencia",),
                language_codes=("en",),
            ),
        ),
    )

    score = score_output(output, expected_case, error=None)

    assert score["exact_match"] is True
    assert score["expected_recall"] == 1.0
    assert score["missing_details"] == []
    assert score["unmatched_candidates"] == []


def test_score_output_matches_split_expected_candidate() -> None:
    output = {
        "nothing_durable": False,
        "candidate_count": 2,
        "candidates": [
            {
                "kind": "evidence",
                "subject_scope": "user",
                "canonical_text": "L'utilisateur prefere les cafes calmes.",
                "source_span": "cafes calmes",
                "support_kind": "direct",
                "preserve_verbatim": False,
                "language_codes": ["fr"],
                "temporal_type": "permanent",
                "valid_from_iso": None,
            },
            {
                "kind": "evidence",
                "subject_scope": "user",
                "canonical_text": "L'utilisateur prefere les cafes avec lumiere naturelle.",
                "source_span": "lumiere naturelle",
                "support_kind": "direct",
                "preserve_verbatim": False,
                "language_codes": ["fr"],
                "temporal_type": "permanent",
                "valid_from_iso": None,
            },
        ],
    }
    case = load_cases(Path("benchmarks/memory_extraction_cards/cases.jsonl"))[0]
    expected_case = case.__class__(
        case_id="cafes",
        message="Je prefere les cafes calmes avec beaucoup de lumiere naturelle.",
        expected_candidates=(
            ExpectedCandidate(
                label="quiet_bright_cafes",
                kind="evidence",
                scope="user",
                must_include=("cafes",),
                any_include_groups=(("calmes", "quiet"), ("lumiere", "light")),
                language_codes=("fr",),
            ),
        ),
    )

    score = score_output(output, expected_case, error=None)

    assert score["exact_match"] is True
    assert score["expected_recall"] == 1.0
    assert score["matched_candidates"][0]["candidate_indices"] == [0, 1]
    assert score["unmatched_candidates"] == []


def test_score_output_matches_accent_insensitive_terms() -> None:
    output = {
        "nothing_durable": False,
        "candidate_count": 1,
        "candidates": [
            {
                "kind": "contract_signal",
                "subject_scope": "user",
                "canonical_text": "El usuario prefiere respuestas técnicas sin emojis.",
                "source_span": "respuestas tecnicas",
                "support_kind": "direct",
                "preserve_verbatim": False,
                "language_codes": ["es"],
                "temporal_type": "permanent",
                "valid_from_iso": None,
            }
        ],
    }
    case = load_cases(Path("benchmarks/memory_extraction_cards/cases.jsonl"))[0]
    expected_case = case.__class__(
        case_id="no_emojis",
        message="Por favor, no uses emojis en respuestas tecnicas.",
        expected_candidates=(
            ExpectedCandidate(
                label="no_emojis_technical",
                must_include=(),
                kind="contract_signal",
                scope="user",
                any_include_groups=(("emoji", "emojis"), ("tecnicas", "technical")),
            ),
        ),
    )

    score = score_output(output, expected_case, error=None)

    assert score["exact_match"] is True
    assert score["expected_recall"] == 1.0


def test_score_output_explains_missing_candidate() -> None:
    output = {
        "nothing_durable": False,
        "candidate_count": 1,
        "candidates": [
            {
                "kind": "state_update",
                "subject_scope": "user",
                "canonical_text": "User is in Valencia.",
                "source_span": "current city is Valencia",
                "support_kind": "direct",
                "preserve_verbatim": False,
                "language_codes": ["en"],
                "temporal_type": "ephemeral",
                "valid_from_iso": None,
            }
        ],
    }
    expected_case = load_cases(Path("benchmarks/memory_extraction_cards/cases.jsonl"))[0].__class__(
        case_id="city",
        message="My current city is Valencia.",
        expected_candidates=(
            ExpectedCandidate(
                label="city",
                kind="evidence",
                scope="user",
                must_include=("valencia",),
                temporal_type="bounded",
                language_codes=("en",),
            ),
        ),
    )

    score = score_output(output, expected_case, error=None)

    assert score["exact_match"] is False
    assert score["missing_labels"] == ["city"]
    reasons = score["missing_details"][0]["candidate_checks"][0]["reasons"]
    assert "kind:state_update!=evidence" in reasons
    assert "temporal_type:ephemeral!=bounded" in reasons
