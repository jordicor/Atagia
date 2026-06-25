from pathlib import Path

from benchmarks.memory_extraction_cards.compare import (
    CardResult,
    CandidateDraft,
    ExpectedCandidate,
    assemble_card_result,
    load_cases,
    normalize_result,
    parse_candidate_card_output,
    parse_coverage_members_card_output,
    parse_evidence_card_output,
    parse_kind_scope_card_output,
    parse_temporal_card_output,
    score_output,
)
from atagia.memory.extraction_cards import _CARD_SYSTEM_PROMPTS


_LINE_ONLY_CARD_SYSTEM_PROMPT = (
    "Extract durable memory as plain-text card lines. "
    "Write only the requested lines. No JSON. No explanation."
)
_COVERAGE_MEMBERS_CARD_SYSTEM_PROMPT = (
    "Extract durable memory in the exact format the task shows. "
    "Write only the requested lines, and write nothing for a candidate the task "
    "resolves as having none. No explanation."
)


def test_memory_extraction_card_system_prompts_are_card_specific() -> None:
    assert _CARD_SYSTEM_PROMPTS["coverage_members"] == (
        _COVERAGE_MEMBERS_CARD_SYSTEM_PROMPT
    )
    for card_name in (
        "candidate",
        "kind_scope",
        "evidence",
        "index",
        "temporal",
        "belief",
    ):
        assert _CARD_SYSTEM_PROMPTS[card_name] == _LINE_ONLY_CARD_SYSTEM_PROMPT


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


def test_enrichment_parsers_accept_line_formats() -> None:
    kind_scope, kind_malformed = parse_kind_scope_card_output(
        "cand_001 evidence user 0.91"
    )
    evidence, evidence_malformed = parse_evidence_card_output(
        "cand_001 direct true en,es | locker code is 7426"
    )
    temporal, temporal_malformed = parse_temporal_card_output(
        "cand_001 event_triggered 2026-06-16T00:00:00+00:00 2026-06-16T23:59:59+00:00"
    )

    assert kind_malformed == 0
    assert evidence_malformed == 0
    assert temporal_malformed == 0
    assert kind_scope["cand_001"]["kind"] == "evidence"
    assert evidence["cand_001"]["preserve_verbatim"] is True
    assert evidence["cand_001"]["language_codes"] == ("en", "es")
    assert temporal["cand_001"]["temporal_type"] == "event_triggered"


def test_coverage_members_card_parser_handles_json_labels() -> None:
    parsed, malformed = parse_coverage_members_card_output(
        'cand_001 | [{"member_key": "dr. mendez", "display_text": "Dr. Mendez; cardiology, clinic A | room 3"}]\n'
        "cand_002 | []"
    )

    assert malformed == 0
    assert parsed["cand_001"][0].member_key == "dr. mendez"
    assert parsed["cand_001"][0].display_text == "Dr. Mendez; cardiology, clinic A | room 3"
    assert parsed["cand_002"] == []


def test_assemble_cards_to_lean_result_and_normalized_output() -> None:
    candidates = (
        CandidateDraft("cand_001", "User's rollback phrase is RIVER-19-BLUE."),
    )
    result, repairs = assemble_card_result(
        candidates,
        [
            CardResult(
                "kind_scope",
                None,
                {"cand_001": {"kind": "evidence", "subject_scope": "user", "confidence": 0.9}},
                True,
            ),
            CardResult(
                "evidence",
                None,
                {
                    "cand_001": {
                        "support_kind": "direct",
                        "preserve_verbatim": True,
                        "language_codes": ("en",),
                        "source_span": "RIVER-19-BLUE",
                    }
                },
                True,
            ),
            CardResult("index", None, {"cand_001": "User's emergency rollback phrase"}, True),
            CardResult("temporal", None, {}, True),
            CardResult("belief", None, {}, True),
        ],
    )

    assert repairs == []
    output = normalize_result(result)
    row = output["candidates"][0]
    assert row["preserve_verbatim"] is True
    assert row["source_span"] == "RIVER-19-BLUE"
    assert row["index_text"] == "User's emergency rollback phrase"


def test_belief_without_claim_fields_is_downgraded() -> None:
    result, repairs = assemble_card_result(
        (CandidateDraft("cand_001", "User prefers challenge-first planning."),),
        [
            CardResult(
                "kind_scope",
                None,
                {"cand_001": {"kind": "belief", "subject_scope": "user", "confidence": 0.82}},
                True,
            ),
            CardResult("evidence", None, {}, True),
            CardResult("index", None, {}, True),
            CardResult("temporal", None, {}, True),
            CardResult("belief", None, {}, True),
        ],
    )

    assert repairs == ["cand_001: belief_without_claim_fields_downgraded"]
    assert normalize_result(result)["candidates"][0]["kind"] == "evidence"


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
