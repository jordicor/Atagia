"""Offline checks for the current coverage benchmark route."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.memory.coverage_members_card import IDENTITY_PURPOSE, MEMBERS_PURPOSE
from atagia.memory.extraction_cards import CandidateDraft
from atagia.models.schemas_memory import CoverageMember
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    RetryPolicy,
    TransientLLMError,
)
from benchmarks.llm_metrics import LLMCallRecorder
from benchmarks.memory_extraction_cards import coverage_format_compare as bench


class ScriptedCoverageProvider(LLMProvider):
    name = "coverage-benchmark-test"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []
        self.failed_first_list = False

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = request.metadata["purpose"]
        prompt = request.messages[-1].content
        if purpose == MEMBERS_PURPOSE:
            if "Mira sees Dr. Ruiz and Dr. Okafor." in prompt:
                if not self.failed_first_list:
                    self.failed_first_list = True
                    raise TransientLLMError("scripted transient failure")
                output = '"Dr. Ruiz"\n"Dr. Okafor"'
            elif "Mira discussed Dr. Vale." in prompt:
                output = "none"
            else:
                raise AssertionError("Unknown candidate in membership request")
        elif purpose == IDENTITY_PURPOSE:
            if "<member>\nDr. Ruiz\n</member>" in prompt:
                output = "Lucía Ruiz"
            elif "<member>\nDr. Okafor\n</member>" in prompt:
                output = "Dr. Okafor"
            else:
                raise AssertionError("Unknown member in identity request")
        else:
            raise AssertionError(f"Unexpected purpose: {purpose}")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output,
            usage={"input_tokens": 12, "output_tokens": 5},
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used")


@pytest.mark.asyncio
async def test_current_variant_runs_production_decisions_and_records_all_attempts() -> None:
    provider = ScriptedCoverageProvider()
    retry = RetryPolicy(
        attempts=2,
        base_delay_seconds=0,
        max_delay_seconds=0,
        jitter_fraction=0,
    )
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        retry_policy=retry,
        extraction_retry_policy=retry,
        structured_output_retry_attempts=0,
    )
    recorder = LLMCallRecorder()
    bench.install_provider_attempt_recorder(client, recorder)
    case = bench.CoverageCase(
        case_id="known_members_and_mention",
        message=(
            "Mira sees Dr. Ruiz (Lucía Ruiz) and Dr. Okafor. "
            "Dr. Vale was only discussed."
        ),
        mode="general_qa",
        candidates=(
            CandidateDraft("cand_001", "Mira sees Dr. Ruiz and Dr. Okafor."),
            CandidateDraft("cand_002", "Mira discussed Dr. Vale."),
        ),
        gold_members={
            "cand_001": (
                CoverageMember(member_key="lucía ruiz", display_text="Dr. Ruiz"),
                CoverageMember(member_key="dr. okafor", display_text="Dr. Okafor"),
            ),
            "cand_002": (),
        },
    )

    with recorder.context(
        benchmark="coverage_format_compare",
        model="test-model",
        variant="current",
    ):
        row = await bench.run_trial(
            client=client,
            case=case,
            variant="current",
            model="test-model",
            repetition=1,
            include_examples=False,
            trial_timeout_seconds=2,
        )

    assert row["error"] is None
    assert row["score"]["all_candidates_exact"] is True
    assert row["score"]["empty_correct_count"] == 1
    assert Counter(request.metadata["purpose"] for request in provider.requests) == {
        MEMBERS_PURPOSE: 3,
        IDENTITY_PURPOSE: 2,
    }
    assert all(
        "<candidate>" in request.messages[-1].content
        and "<candidates>" not in request.messages[-1].content
        for request in provider.requests
    )

    now = datetime.now(timezone.utc)
    report = bench.summarize_run(
        [row],
        recorder=recorder,
        models=("test-model",),
        variants=("current",),
        repetitions=1,
        started_at=now,
        finished_at=now,
        cases_path=Path("synthetic.jsonl"),
    )
    summary = report["llm_call_summary"]
    assert summary["total_calls"] == 5
    assert summary["failed_calls"] == 1
    assert summary["by_purpose"][MEMBERS_PURPOSE]["calls"] == 3
    assert summary["by_purpose"][IDENTITY_PURPOSE]["calls"] == 2
    assert report["results"]["test-model::current"]["provider_attempts"] == 5


def test_only_challenger_parsers_accept_compound_output() -> None:
    assert bench.run_self_test() == 0
    with pytest.raises(ValueError, match="production executor"):
        bench.parse_variant_output("current", "none", ("cand_001",))
