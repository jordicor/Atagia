"""Boundary validation and single-source-of-truth tests for
``override_retrieval_params``.

CS-1.2: unknown/typo'd override keys were silently ignored, so a run could vary
a knob the retrieval pipeline never reads. These tests pin the recognized-keys
constant to the pipeline consumption sites (no drift) and assert the
AblationConfig boundary fails fast on unrecognized keys.

The same boundary rejects unusable VALUES. Clamping them was the value-level
version of the same defect: a run recorded ``privacy_ceiling=99`` while the
engine ran 3. With rejection in place, recorded equals requested by
construction, so these tests assert the rejection and the ranges that drive it
-- re-deriving those ranges from the models that own the constraints, so the
two can never drift apart.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import annotated_types
import pytest
from pydantic import BaseModel, ValidationError

from atagia.models.schemas_memory import (
    RetrievalParams,
    RetrievalPlan,
    RetrievalProfileManifest,
)
from atagia.models.schemas_replay import (
    _OVERRIDE_RETRIEVAL_PARAM_BOUNDS,
    LLM_COVERAGE_MAX_SUBQUERIES,
    RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS,
    AblationConfig,
)

_PIPELINE_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "atagia"
    / "services"
    / "retrieval_pipeline.py"
)


def _override_param_keys_read_by_pipeline() -> set[str]:
    """Extract every ``override_retrieval_params`` key the pipeline reads.

    Two access shapes appear in ``retrieval_pipeline.py``:

    * Explicit string literals -- ``override_params["key"]`` subscripts and
      ``"key" (not) in override_params`` membership tests.
    * Dynamic field iteration -- ``for field_name in RetrievalParams.model_fields``
      inside ``_override_policy``, which applies every ``RetrievalParams`` field.

    The union of both is exactly what the consumption sites read, and it must
    equal ``RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS``.
    """
    tree = ast.parse(_PIPELINE_SOURCE.read_text(encoding="utf-8"))
    literal_keys: set[str] = set()

    def _is_override_params(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == "override_params"

    for node in ast.walk(tree):
        # override_params["key"]
        if isinstance(node, ast.Subscript) and _is_override_params(node.value):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                literal_keys.add(key.value)
        # "key" in override_params  /  "key" not in override_params
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant):
            left_value = node.left.value
            if isinstance(left_value, str):
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.In, ast.NotIn)) and _is_override_params(
                        comparator
                    ):
                        literal_keys.add(left_value)
        # override_params.get("key"[, default]) — the shape the coverage
        # knobs and the two prompt-section budgets use, so this branch is
        # load-bearing, not speculative.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_override_params(node.func.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            literal_keys.add(node.args[0].value)

    return literal_keys | set(RetrievalParams.model_fields)


def test_constant_matches_pipeline_consumption_sites_no_drift() -> None:
    assert _override_param_keys_read_by_pipeline() == set(
        RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS
    )


def test_constant_has_expected_twelve_keys() -> None:
    # The Fase 0 finding (cs0_3_experimental_truth) counted 13 recognized keys.
    # graph_hops has since been removed: it had no reader anywhere in the
    # engine, so accepting it as an override let a run vary a knob nothing
    # consumed -- the exact defect the recognized-keys constant exists to stop.
    assert len(RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS) == 12
    assert "graph_hops" not in RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS
    assert "graph_hops" not in RetrievalParams.model_fields


def _declared_bound(model: type[BaseModel], field_name: str) -> tuple[int, int | None]:
    """Re-derive one field's inclusive integer bounds from its own constraints."""
    minimum: int | None = None
    maximum: int | None = None
    for constraint in model.model_fields[field_name].metadata:
        if isinstance(constraint, annotated_types.Ge):
            minimum = int(constraint.ge)
        elif isinstance(constraint, annotated_types.Gt):
            minimum = int(constraint.gt) + 1
        elif isinstance(constraint, annotated_types.Le):
            maximum = int(constraint.le)
        elif isinstance(constraint, annotated_types.Lt):
            maximum = int(constraint.lt) - 1
    assert minimum is not None, f"{model.__name__}.{field_name} declares no lower bound"
    return (minimum, maximum)


@pytest.mark.parametrize(
    ("override_key", "model", "field_name"),
    [
        # The four keys applied straight onto RetrievalParams.
        ("fts_limit", RetrievalParams, "fts_limit"),
        ("vector_limit", RetrievalParams, "vector_limit"),
        ("rerank_top_k", RetrievalParams, "rerank_top_k"),
        ("final_context_items", RetrievalParams, "final_context_items"),
        # The keys applied onto the retrieval plan.
        ("max_candidates", RetrievalPlan, "max_candidates"),
        ("max_context_items", RetrievalPlan, "max_context_items"),
        ("privacy_ceiling", RetrievalPlan, "privacy_ceiling"),
        # The prompt-section budgets, bounded by the manifest that declares them.
        ("context_budget_tokens", RetrievalProfileManifest, "context_budget_tokens"),
        (
            "transcript_budget_tokens",
            RetrievalProfileManifest,
            "transcript_budget_tokens",
        ),
    ],
)
def test_validated_range_matches_the_model_that_owns_the_constraint(
    override_key: str,
    model: type[BaseModel],
    field_name: str,
) -> None:
    """The boundary must reject exactly what the target model would reject.

    A looser range here would let an override build a model that violates its
    own Field bounds; a tighter one would reject a value the engine can honor.
    """
    assert _OVERRIDE_RETRIEVAL_PARAM_BOUNDS[override_key] == _declared_bound(
        model, field_name
    )


def test_subquery_range_matches_the_pipeline_default_bound() -> None:
    assert _OVERRIDE_RETRIEVAL_PARAM_BOUNDS["llm_coverage_max_subqueries"] == (
        1,
        LLM_COVERAGE_MAX_SUBQUERIES,
    )


def test_every_recognized_key_has_a_validated_domain() -> None:
    """No key may be accepted without a declared range or flag domain."""
    assert set(RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS) == set(
        _OVERRIDE_RETRIEVAL_PARAM_BOUNDS
    ) | {"allow_private_sensitivity"}


@pytest.mark.parametrize(
    ("requested", "expected_range"),
    [
        # The measured CS-1.2 divergences: the trace said 99 while the engine
        # ran 3. Neither value is honorable, so neither is accepted.
        ({"privacy_ceiling": 99}, "0..3"),
        ({"privacy_ceiling": -7}, "0..3"),
        ({"llm_coverage_max_subqueries": 99}, "1..3"),
        ({"llm_coverage_max_subqueries": 0}, "1..3"),
        ({"context_budget_tokens": -5}, ">= 1"),
        ({"transcript_budget_tokens": -50}, ">= 1"),
        ({"max_candidates": -1}, ">= 0"),
        ({"max_context_items": 0}, ">= 1"),
        ({"vector_limit": -3}, ">= 0"),
        ({"final_context_items": 0}, ">= 1"),
        ({"llm_coverage_candidate_limit": 0}, ">= 1"),
        # The fail-fast hole model_copy(update=...) used to leave open: an
        # override could build a RetrievalParams violating its own gt=0.
        ({"rerank_top_k": 0}, ">= 1"),
        ({"fts_limit": -1}, ">= 0"),
    ],
)
def test_out_of_range_value_is_rejected_naming_key_value_and_range(
    requested: dict[str, object],
    expected_range: str,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        AblationConfig(override_retrieval_params=requested)
    message = str(excinfo.value)
    key, value = next(iter(requested.items()))
    assert key in message
    assert str(value) in message
    assert expected_range in message


def test_every_out_of_range_value_is_named_in_one_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        AblationConfig(
            override_retrieval_params={
                "privacy_ceiling": 99,
                "rerank_top_k": 0,
                "fts_limit": 24,
            }
        )
    message = str(excinfo.value)
    assert "privacy_ceiling=99" in message
    assert "rerank_top_k=0" in message
    # The honorable key is not reported as an offender.
    assert "fts_limit=24" not in message


@pytest.mark.parametrize(
    "requested",
    [
        # bool is an int subclass, so a flag must not pass as a bounded integer.
        {"rerank_top_k": True},
        {"privacy_ceiling": False},
        # Nor may a bounded integer be supplied as text the engine would not
        # compare numerically.
        {"fts_limit": "40"},
        {"max_candidates": 3.5},
        # And the one flag knob must be an actual flag.
        {"allow_private_sensitivity": 1},
        {"allow_private_sensitivity": "yes"},
    ],
)
def test_wrongly_typed_value_is_rejected(requested: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as excinfo:
        AblationConfig(override_retrieval_params=requested)
    assert next(iter(requested)) in str(excinfo.value)


@pytest.mark.parametrize(
    "requested",
    [
        {"privacy_ceiling": 0},
        {"privacy_ceiling": 3},
        {"llm_coverage_max_subqueries": 1},
        {"llm_coverage_max_subqueries": LLM_COVERAGE_MAX_SUBQUERIES},
        {"max_candidates": 0},
        {"vector_limit": 0},
        {"fts_limit": 40},
        {"rerank_top_k": 9},
        {"context_budget_tokens": 1},
        {"allow_private_sensitivity": True},
        {"allow_private_sensitivity": False},
    ],
)
def test_in_range_values_pass_through_unchanged(requested: dict[str, object]) -> None:
    """Applied equals requested: no normalization step sits between them."""
    ablation = AblationConfig(override_retrieval_params=requested)
    assert ablation.override_retrieval_params == requested


def test_all_recognized_keys_pass_validation() -> None:
    honorable: dict[str, Any] = {
        key: (
            True
            if key == "allow_private_sensitivity"
            else max(1, _OVERRIDE_RETRIEVAL_PARAM_BOUNDS[key][0])
        )
        for key in RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS
    }
    for key, value in honorable.items():
        ablation = AblationConfig(override_retrieval_params={key: value})
        assert ablation.override_retrieval_params == {key: value}


def test_none_and_empty_override_params_are_allowed() -> None:
    assert AblationConfig().override_retrieval_params is None
    assert AblationConfig(override_retrieval_params={}).override_retrieval_params == {}


def test_historical_unknown_key_fails_fast_naming_the_key() -> None:
    # The exact key a 319-run sweep varied while the pipeline read it nowhere.
    with pytest.raises(ValidationError) as excinfo:
        AblationConfig(
            override_retrieval_params={"retrieval_side_llm_call_budget": 3}
        )
    message = str(excinfo.value)
    assert "retrieval_side_llm_call_budget" in message
    # The error lists the recognized set to guide the fix.
    assert "fts_limit" in message
    assert "rerank_top_k" in message


def test_near_miss_typo_fails_fast() -> None:
    with pytest.raises(ValidationError) as excinfo:
        AblationConfig(override_retrieval_params={"rerank_topk": 3})
    assert "rerank_topk" in str(excinfo.value)


def test_multiple_unknown_keys_all_named_valid_key_excluded() -> None:
    with pytest.raises(ValidationError) as excinfo:
        AblationConfig(
            override_retrieval_params={
                "bogus_one": 1,
                "bogus_two": 2,
                "fts_limit": 10,
            }
        )
    message = str(excinfo.value)
    offending = message.split("key(s):")[1].split("Recognized keys:")[0]
    assert "bogus_one" in offending
    assert "bogus_two" in offending
    # The valid key is not reported as an offender.
    assert "fts_limit" not in offending


def test_valid_multi_key_override_preserves_all_values() -> None:
    override = {
        "fts_limit": 24,
        "rerank_top_k": 3,
        "final_context_items": 12,
        "privacy_ceiling": 2,
        "allow_private_sensitivity": True,
    }
    ablation = AblationConfig(override_retrieval_params=override)
    assert ablation.override_retrieval_params == override


def test_model_validate_from_dict_rejects_unknown_key() -> None:
    # Admin replay / serialized configs rehydrate AblationConfig via
    # model_validate; that boundary must fail fast too.
    with pytest.raises(ValidationError) as excinfo:
        AblationConfig.model_validate(
            {"override_retrieval_params": {"retrieval_side_llm_call_budget": 3}}
        )
    assert "retrieval_side_llm_call_budget" in str(excinfo.value)
