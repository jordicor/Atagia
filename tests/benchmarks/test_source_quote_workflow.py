"""Run every workflow arm through production builders, parsers and adapters."""

import json

import httpx
import pytest

from atagia.services.llm_client import LLMClient, LLMCompletionResponse
from atagia.services.providers.openrouter import OpenRouterProvider
from atagia.services.providers.typesafe import TypeSafeProvider
from benchmarks.source_quote_selection import compact_run, workflow


def test_resumption_requires_diagnosis_of_exact_failed_result(tmp_path):
    row = {
        "slot": "acceptance:case:arm:1",
        "status": "technical_failure",
        "error": "Invalid distribution",
    }
    rows = {row["slot"]: row}
    with pytest.raises(ValueError, match="coordinator diagnosis"):
        workflow.validate_prior_failures(tmp_path, rows)
    acknowledgement = tmp_path / "acknowledged_failures.json"
    acknowledgement.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    workflow.validate_prior_failures(tmp_path, rows)
    assert rows[row["slot"]]["status"] == "technical_failure"
    changed = {row["slot"]: {**row, "error": "Different failure"}}
    with pytest.raises(ValueError, match="coordinator diagnosis"):
        workflow.validate_prior_failures(tmp_path, changed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("evaluate", "arm"),
    [(workflow.evaluate, arm) for arm in workflow.ARMS]
    + [(compact_run.evaluate, arm) for arm in compact_run.ARMS],
)
async def test_complete_evidence_workflow_is_wired_before_live_smoke(evaluate, arm):
    class Capture(OpenRouterProvider):
        def __init__(self):
            super().__init__(
                "unused",
                site_url="https://example.invalid",
                app_name="offline",
                client=object(),
            )
            self.calls = []

        async def complete(self, request):
            self.calls.append(self._completion_kwargs(request, stream=False))
            content = {
                "memory_extraction_evidence_support_card": "direct",
                "memory_extraction_preserve_verbatim_card": "yes",
                "memory_extraction_candidate_language_card": "en",
                "memory_extraction_source_reference_card": "r2 r4",
            }[request.metadata["purpose"]]
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text=content
            )

    native_requests = []

    async def respond(request):
        body = json.loads(request.content)
        native_requests.append(body)
        answers = {}
        for key, question in body["questions"].items():
            choice = "r2" if key.endswith(".start") else "r4"
            answers[key] = {
                "type": "choice",
                "choice": choice,
                "confidence": 1.0,
                "probabilities": {
                    option: float(option == choice) for option in question["criteria"]
                },
            }
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 40, "output_tokens": 0},
            },
        )

    source = "Keep AX-12."
    case = {
        "case_id": "offline",
        "source_text": source,
        "recent_messages": [
            {"role": "assistant", "content": "Which reference should I keep?"}
        ],
        "candidates": [
            {
                "candidate_id": "cand_001",
                "canonical_text": "The reference is AX-12.",
                "expected_ranges": [[5, 10]],
                "required_ranges": [[5, 10]],
                "allowed_range": [0, 11],
            }
        ],
    }
    provider = Capture()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = LLMClient(
            providers=[provider, TypeSafeProvider("unused", client=http)]
        )
        try:
            result = await evaluate(client, case, arm)
        finally:
            await client.aclose()
    assert result["quotes"] == {"cand_001": "AX-12"}
    assert result["grades"][0]["adequate"] is True
    assert result["evidence"]["cand_001"]["preserve_verbatim"] is True
    assert result["evidence"]["cand_001"]["language_codes"] == ("en",)
    assert len(provider.calls) == (3 if arm.startswith("jev_") else 4)
    for wire in provider.calls:
        assert wire["extra_body"]["reasoning"]["effort"] == "none"
        assert "temperature" not in wire
        assert "Which reference should I keep?" in wire["messages"][-1]["content"]
    assert len(native_requests) == (1 if arm.startswith("jev_") else 0)
    if native_requests:
        assert (
            "Which reference should I keep?"
            in native_requests[0]["state"][-1]["content"]
        )
