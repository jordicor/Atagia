from __future__ import annotations

from benchmarks.applicability_cards.compare import (
    _DEFAULT_CASES_PATH,
    _estimate_cost_usd,
    expand_cases,
    load_cases,
    score_output,
)


def test_applicability_card_cases_load() -> None:
    cases = load_cases(_DEFAULT_CASES_PATH)

    assert len(cases) == 8
    assert cases[0].case_id == "slot_current_drone_location"
    assert cases[0].candidates[0]["id"] == "mem_drone_current"


def test_applicability_card_cases_expand_deterministically() -> None:
    cases = expand_cases(load_cases(_DEFAULT_CASES_PATH), 50)

    assert len(cases) == 50
    assert cases[8].case_id == "synth_000_current_city"
    assert cases[-1].case_id == "synth_041_fr_code"
    assert len({case.case_id for case in cases}) == 50


def test_score_output_accepts_expected_top_and_useful_hits() -> None:
    case = load_cases(_DEFAULT_CASES_PATH, limit=1)[0]

    score = score_output(
        [
            {"memory_id": "mem_drone_current", "resolved_date": None},
            {"memory_id": "mem_robotics_trip", "resolved_date": None},
        ],
        case,
    )

    assert score["exact_match"] is True
    assert score["top_hit"] is True
    assert score["expected_useful_recall"] == 1.0


def test_score_output_rejects_expected_drop_in_top3() -> None:
    case = load_cases(_DEFAULT_CASES_PATH, limit=1)[0]

    score = score_output(
        [
            {"memory_id": "mem_workshop_pigment", "resolved_date": None},
            {"memory_id": "mem_drone_current", "resolved_date": None},
        ],
        case,
    )

    assert score["exact_match"] is False
    assert score["expected_drop_top3_hits"] == ["mem_workshop_pigment"]


def test_estimate_cost_uses_cached_minimax_input_rate() -> None:
    cost = _estimate_cost_usd(
        "minimax/MiniMax-M3",
        {
            "input_tokens": 1000,
            "cached_input_tokens": 200,
            "output_tokens": 100,
        },
    )

    assert cost == (800 * 0.30 + 200 * 0.06 + 100 * 1.20) / 1_000_000


def test_unprocessed_date_expectations_do_not_count_as_full_match() -> None:
    case = next(
        case for case in load_cases(_DEFAULT_CASES_PATH)
        if any(value is not None for value in (case.expected_resolved_dates or {}).values())
    )
    memory_ids = list(dict.fromkeys([*case.expected_top_ids, *case.expected_useful_ids]))
    score = score_output(
        [{"memory_id": memory_id, "resolved_date": None, "date_resolution_status": "unprocessed"}
         for memory_id in memory_ids],
        case,
    )
    assert score["ranking_match"] is True
    assert score["exact_match"] is False
    assert score["date_evaluation_status"] == "unavailable_unprocessed"
    assert all(value is None for value in score["expected_date_matches"].values())
