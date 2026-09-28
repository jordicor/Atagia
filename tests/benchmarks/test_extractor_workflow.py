"""Offline checks for the frozen full-extractor comparison."""

from __future__ import annotations

import json

import pytest

from atagia.core.source_references import SourceReferenceCatalog
from atagia.models.schemas_decisions import ChoiceAnswer, ScoreAnswer
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.model_resolution import DEFAULT_FINITE_DECISION_MODEL
from benchmarks.extractor_workflow import run


class ScriptedProvider(LLMProvider):
    name = "openrouter"

    def __init__(
        self, source: str, *, no_memory: bool = False, bad_reference: bool = False
    ):
        self.source = source
        self.no_memory = no_memory
        self.bad_reference = bad_reference
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = request.metadata.get("purpose")
        last = SourceReferenceCatalog(self.source).anchors[-1].reference_id
        output = {
            "memory_extraction_candidate_card": (
                "none"
                if self.no_memory
                else "cand_001 | The user's default editor is Zed."
            ),
            "memory_extraction_kind_card": "evidence",
            "memory_extraction_scope_card": "user",
            "memory_extraction_confidence_card": "0.83",
            "memory_extraction_evidence_support_card": "direct",
            "memory_extraction_preserve_verbatim_card": "yes",
            "memory_extraction_candidate_language_card": "en",
            "memory_extraction_source_reference_card": f"r1 {'r999' if self.bad_reference else last}",
            "memory_extraction_index_card": "cand_001 | default editor Zed",
            "memory_extraction_temporal_type_card": "none",
            "memory_extraction_coverage_members_card": "none",
            "intent_classifier_explicit": json.dumps(
                {
                    "is_explicit": True,
                    "reasoning": "The source directly states this fact.",
                }
            ),
        }.get(purpose)
        if output is None:
            raise AssertionError(f"Unexpected production call: {purpose}")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output,
            usage={"input_tokens": 20, "output_tokens": 8},
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embedding is disabled")


def _client(provider: ScriptedProvider) -> LLMClient:
    return LLMClient(
        providers=[provider],
        structured_output_retry_attempts=0,
        max_concurrent_requests_per_provider=2,
    )


class ScriptedTypeSafeProvider(LLMProvider):
    name = "typesafe"
    supports_choices = True
    supports_scores = True

    def __init__(self, *, confidence_score: float = 3.32) -> None:
        self.requests: list[LLMCompletionRequest] = []
        self.confidence_score = confidence_score

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        answers = {}
        values_by_purpose = {
            "memory_extraction_kind_card": "evidence",
            "memory_extraction_scope_card": "user",
            "memory_extraction_temporal_type_card": "none",
        }
        purpose = str(request.metadata["purpose"])
        if purpose == "memory_extraction_confidence_card":
            lower = int(self.confidence_score)
            fraction = self.confidence_score - lower
            probabilities = {str(index): 0.0 for index in range(5)}
            probabilities[str(lower)] = 1.0 - fraction
            if fraction:
                probabilities[str(lower + 1)] = fraction
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                score_answers={
                    key: ScoreAnswer(
                        type="score",
                        score=self.confidence_score,
                        legend={str(index): level for index, level in enumerate(question.criteria)},
                        probabilities=probabilities,
                        confidence=0.02,
                    )
                    for key, question in (request.score_questions or {}).items()
                },
                usage={"input_tokens": 20, "output_tokens": 0},
            )
        if purpose not in values_by_purpose:
            raise AssertionError(f"Unexpected TypeSafe purpose: {purpose}")
        for key in request.choice_questions or {}:
            value = values_by_purpose[purpose]
            answers[key] = ChoiceAnswer(
                type="choice",
                name=key,
                choice=value,
                probabilities={value: 1.0},
                confidence=1.0,
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            choice_answers=answers,
            usage={"input_tokens": 20, "output_tokens": 0},
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embedding is disabled")


@pytest.mark.asyncio
async def test_no_memory_runs_real_database_without_persistence() -> None:
    source = "Thanks, that answers my question."
    provider = ScriptedProvider(source, no_memory=True)
    result = await run._run_slot(
        _client(provider),
        {
            "source_text": source,
            "role": "user",
            "recent_messages": [],
        },
        "luna56",
    )
    assert result["raw_extraction"]["nothing_durable"] is True
    assert result["persisted"] == []
    assert result["database_rows"]["memory_objects"] == []
    assert [request.metadata["purpose"] for request in provider.requests] == [
        "memory_extraction_candidate_card"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["luna56", "luna6"])
async def test_durable_fact_persists_exact_source_reference_and_fixed_evidence(
    arm: str,
) -> None:
    source = "My default editor is Zed."
    provider = ScriptedProvider(source)
    result = await run._run_slot(
        _client(provider),
        {
            "source_text": source,
            "role": "user",
            "recent_messages": [],
        },
        arm,
    )
    assert result["raw_extraction"]["nothing_durable"] is False
    assert result["source_reference_checks"][0]["literal_quote"] == source
    assert result["database_rows"]["memory_objects"]
    spans = result["database_rows"]["memory_evidence_spans"]
    assert any(
        span["quote_text"] == source
        and span["char_start"] == 0
        and span["char_end"] == len(source)
        for span in spans
    )
    assert all(check["quote_matches"] for check in result["source_reference_checks"])
    requests = {request.metadata["purpose"]: request for request in provider.requests}
    assert requests["memory_extraction_candidate_card"].model == run.ARMS[
        arm
    ].removeprefix("openrouter/")
    for purpose in (
        "memory_extraction_kind_card",
        "memory_extraction_scope_card",
        "memory_extraction_confidence_card",
    ):
        assert requests[purpose].model == run.ARMS[arm].removeprefix("openrouter/")
    assert requests[
        "memory_extraction_source_reference_card"
    ].model == run.EVIDENCE_MODEL.removeprefix("openrouter/")
    assert result["chunk_count"] == 1


@pytest.mark.asyncio
async def test_invalid_source_coordinates_fail_fast() -> None:
    source = "My default editor is Zed."
    with pytest.raises(ValueError, match="source reference"):
        await run._run_slot(
            _client(ScriptedProvider(source, bad_reference=True)),
            {
                "source_text": source,
                "role": "user",
                "recent_messages": [],
            },
            "luna56",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "expected_status"),
    [(3.32, "active"), (1.2, "review_required")],
)
async def test_native_classification_score_controls_persisted_status(
    score: float, expected_status: str,
) -> None:
    source = "My default editor is Zed."
    openrouter = ScriptedProvider(source)
    typesafe = ScriptedTypeSafeProvider(confidence_score=score)
    client = LLMClient(
        providers=[openrouter, typesafe],
        structured_output_retry_attempts=0,
        max_concurrent_requests_per_provider=2,
    )
    result = await run._run_slot(
        client,
        {
            "source_text": source,
            "role": "user",
            "recent_messages": [],
        },
        "luna6_jev_classification",
    )
    assert result["database_rows"]["memory_objects"]
    memory = result["database_rows"]["memory_objects"][0]
    assert memory["confidence"] == pytest.approx(score / 4)
    assert memory["status"] == expected_status
    assert len(typesafe.requests) == 4
    assert all(
        set(request.choice_questions or request.score_questions or {}) == {"cand_001"}
        for request in typesafe.requests
    )
    assert {request.metadata["purpose"] for request in typesafe.requests} == {
        "memory_extraction_kind_card",
        "memory_extraction_scope_card",
        "memory_extraction_confidence_card",
        "memory_extraction_temporal_type_card",
    }
    assert all(
        request.model == (
            DEFAULT_FINITE_DECISION_MODEL
            if request.metadata["purpose"] == "memory_extraction_temporal_type_card"
            else run.JEV_MODEL
        ).removeprefix("typesafe/")
        for request in typesafe.requests
    )
    assert all(
        request.metadata.get("purpose") not in {
            "memory_extraction_kind_card",
            "memory_extraction_scope_card",
            "memory_extraction_confidence_card",
            "memory_extraction_temporal_type_card",
        }
        for request in openrouter.requests
    )
    evidence = [
        request
        for request in openrouter.requests
        if request.metadata.get("purpose") == "memory_extraction_source_reference_card"
    ]
    assert len(evidence) == 1
    assert evidence[0].model == run.EVIDENCE_MODEL.removeprefix("openrouter/")


def test_freeze_snapshots_real_sources_and_detects_tampering(tmp_path) -> None:
    helper = tmp_path / "budget_helper.py"
    helper.write_text("# frozen test helper\n", encoding="utf-8")
    output = tmp_path / "experiment"
    output.mkdir()
    (output / "coordinator_control.json").write_text("{}", encoding="utf-8")
    manifest = run.prepare(output, helper)
    assert len([slot for slot in manifest["slots"] if slot["phase"] == "smoke"]) == len(
        run.ARMS
    ) * len(run.load_smoke_cases())
    assert len(
        [slot for slot in manifest["slots"] if slot["phase"] == "evaluation"]
    ) == 10 * len(run.ARMS) * len(run.load_cases())
    assert run.verify_freeze(output, helper) == manifest
    assert manifest["slots"][0]["arm"] != manifest["slots"][2]["arm"]
    snapshot = output / "source_snapshot" / "src/atagia/memory/extractor.py"
    snapshot.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source changed"):
        run.verify_freeze(output, helper)


def test_prior_chunk_context_requires_a_real_chunked_source(monkeypatch) -> None:
    monkeypatch.setattr(
        run,
        "load_cases",
        lambda: [
            {
                "case_id": "synthetic_prior",
                "source_text": "A fact.",
                "role": "user",
                "prior_chunk_context": "A prior candidate.",
            }
        ],
    )
    with pytest.raises(ValueError, match="synthetic prior chunk context"):
        run._cases()


def test_technical_failure_requires_matching_acknowledgment(tmp_path) -> None:
    rows = {"one": {"status": "error"}}
    with pytest.raises(ValueError, match="coordinator acknowledgment"):
        run._validate_prior_failures(tmp_path, rows, "freeze")
    (tmp_path / "failure_ack.json").write_text(
        json.dumps(
            {
                "freeze_sha256": "freeze",
                "acknowledged_slots": ["one"],
                "reason": "Coordinator reviewed the terminal failure and journal.",
            }
        ),
        encoding="utf-8",
    )
    run._validate_prior_failures(tmp_path, rows, "freeze")
    with pytest.raises(ValueError, match="does not match"):
        run._validate_prior_failures(tmp_path, rows, "changed")
