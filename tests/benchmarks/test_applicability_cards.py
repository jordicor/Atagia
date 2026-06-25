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
    assert cases[0].case_id == "slot_current_city"
    assert cases[0].candidates[0]["id"] == "mem_city_current"


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
            {"memory_id": "mem_city_current", "resolved_date": None},
            {"memory_id": "mem_city_trip", "resolved_date": None},
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
            {"memory_id": "mem_art", "resolved_date": None},
            {"memory_id": "mem_city_current", "resolved_date": None},
        ],
        case,
    )

    assert score["exact_match"] is False
    assert score["expected_drop_top3_hits"] == ["mem_art"]


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
