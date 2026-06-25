from __future__ import annotations

from benchmarks.need_detection_cards.__main__ import (
    HYBRID_NAKED_CONFIGS,
    MODEL_SPECS,
    TrialResult,
    _card_request,
    _case_set,
    _deterministic_naked_card_call,
    _merge_naked_cards,
    _naked_card_request,
    _parse_naked_output,
    _resolve_hybrid_card_models,
    _summarize,
    NakedCardCall,
)


def test_need_detection_cards_case_set_has_phase_sizes() -> None:
    cases = _case_set()

    assert len(cases) == 20
    assert len(cases[:12]) == 12
    assert all(case.expected is not None for case in cases)


def test_need_detection_card_requests_have_structured_schema() -> None:
    case = _case_set()[0]
    native_model = MODEL_SPECS["gemini_flash_lite"]
    prompt_json_model = MODEL_SPECS["gpt_oss_20b"]

    for card_name in ("language", "memory_dependence", "exactness", "anchors"):
        request, schema = _card_request(
            case=case,
            card_name=card_name,  # type: ignore[arg-type]
            model=native_model,
        )

        assert request.response_schema
        assert request.metadata["openrouter_native_structured_output"] is True
        assert request.metadata["provider_extra_body"]
        assert schema.model_json_schema()

        prompt_request, prompt_schema = _card_request(
            case=case,
            card_name=card_name,  # type: ignore[arg-type]
            model=prompt_json_model,
        )

        assert prompt_request.response_schema is None
        assert "openrouter_native_structured_output" not in prompt_request.metadata
        assert "Return exactly one raw JSON" in prompt_request.messages[-1].content
        assert prompt_schema.model_json_schema()


def test_naked_card_request_uses_tiny_raw_format() -> None:
    case = _case_set()[0]
    model = MODEL_SPECS["gpt_oss_20b"]

    request = _naked_card_request(case=case, card_name="memory", model=model)

    assert request.response_schema is None
    assert request.max_output_tokens <= 32
    assert "Output only one word" in request.messages[-1].content
    assert request.metadata["purpose"] == "need_detection_naked_memory"


def test_parse_naked_outputs_and_merge_plan() -> None:
    case = _case_set()[1]
    parsed_language, valid_language = _parse_naked_output("language", "es<TAB>es")
    parsed_memory, valid_memory = _parse_naked_output("memory", "personal")
    parsed_exact, valid_exact = _parse_naked_output("exact", "yes")
    parsed_shape, valid_shape = _parse_naked_output("shape", "slot")
    parsed_facets, valid_facets = _parse_naked_output("facets", "location quantity")
    parsed_callback, valid_callback = _parse_naked_output("callback", "non")
    parsed_search_words, valid_search_words = _parse_naked_output(
        "search_words",
        "Ben\napartamento\ndireccion",
    )

    assert all(
        [
            valid_language,
            valid_memory,
            valid_exact,
            valid_shape,
            valid_facets,
            valid_callback,
            valid_search_words,
        ]
    )

    calls = [
        NakedCardCall("language", 0.1, True, parsed=parsed_language),
        NakedCardCall("memory", 0.1, True, parsed=parsed_memory),
        NakedCardCall("exact", 0.1, True, parsed=parsed_exact),
        NakedCardCall("shape", 0.1, True, parsed=parsed_shape),
        NakedCardCall("facets", 0.1, True, parsed=parsed_facets),
        NakedCardCall("callback", 0.1, True, parsed=parsed_callback),
        NakedCardCall("search_words", 0.1, True, parsed=parsed_search_words),
    ]

    merged = _merge_naked_cards(case, calls)

    assert merged["query_language"] == "es"
    assert merged["memory_dependence"] == "personal"
    assert merged["query_type"] == "slot_fill"
    assert merged["exact_recall_needed"] is True
    assert merged["exact_facets"] == ["location", "quantity"]
    assert merged["metrics"]["missing_anchor_terms"] == []


def test_hybrid_config_can_route_search_words_without_llm() -> None:
    config = HYBRID_NAKED_CONFIGS["hybrid_qwen_memory_no_anchor"]
    card_models = _resolve_hybrid_card_models(config)

    assert card_models["memory"] == MODEL_SPECS["qwen30b_a3b"]
    assert card_models["exact"] == MODEL_SPECS["gemini_flash_lite"]
    assert card_models["search_words"] is None

    call = _deterministic_naked_card_call(card_name="search_words")

    assert call.parse_valid is True
    assert call.model_label == "deterministic"
    assert call.parsed == {"anchor_terms": []}
    assert call.cost_usd == 0.0


def test_need_detection_cards_summary_counts_unsafe_metrics() -> None:
    results = [
        TrialResult(
            phase="phase2",
            config_label="cards_all_gpt_oss_20b",
            model_label="gpt_oss_20b",
            trial_kind="card",
            card_name="exactness",
            case_id="case_1",
            iteration=0,
            latency_seconds=1.0,
            schema_valid=True,
            sanity_ok=True,
            tokens_in=100,
            tokens_out=10,
            cost_usd=0.001,
            output={"exact_recall_needed": False},
            metrics={"unsafe_exact_miss": True, "missing_facets": ["date"]},
        ),
        TrialResult(
            phase="phase2",
            config_label="cards_all_gpt_oss_20b",
            model_label="gpt_oss_20b",
            trial_kind="card",
            card_name="memory_dependence",
            case_id="case_1",
            iteration=0,
            latency_seconds=2.0,
            schema_valid=True,
            sanity_ok=False,
            tokens_in=100,
            tokens_out=10,
            cost_usd=0.002,
            output={"memory_dependence": "world"},
            metrics={"unsafe_memory_skip": True},
        ),
    ]

    summary = _summarize(results)[0]

    assert summary.unsafe_exact_misses == 1
    assert summary.unsafe_memory_skips == 1
    assert summary.missing_facet_count == 1
    assert summary.schema_valid_pct == 100.0
    assert summary.sanity_ok_pct == 50.0
    assert summary.estimated_cost_usd == 0.003
