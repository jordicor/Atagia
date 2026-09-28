"""Check that the added model arm preserves the baseline's effective request."""

import pytest

from atagia.services.llm_client import LLMClient, LLMCompletionResponse
from atagia.services.model_profiles import MODEL_PROFILES, ModelProfile
from atagia.services.providers.openrouter import OpenRouterProvider
from benchmarks.source_quote_selection import run as original
from benchmarks.source_quote_selection.run_luna6 import MODEL, ModelOnlyClient


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["current_evidence", "luna_reference"])
async def test_only_the_model_changes_on_the_wire(monkeypatch, arm):
    class Capture(OpenRouterProvider):
        def __init__(self):
            super().__init__(
                "unused",
                site_url="https://example.invalid",
                app_name="offline-test",
                client=object(),
            )
            self.requests = []

        async def complete(self, request):
            self.requests.append(self._completion_kwargs(request, stream=False))
            output = (
                {
                    "memory_extraction_evidence_support_card": "direct",
                    "memory_extraction_preserve_verbatim_card": "no",
                    "memory_extraction_candidate_language_card": "en",
                    "memory_extraction_source_reference_card": "r1 r3",
                }[request.metadata["purpose"]]
                if arm == "current_evidence"
                else "cand_001 | r1 r3"
            )
            return LLMCompletionResponse(
                provider="openrouter", model=request.model, output_text=output
            )

    monkeypatch.setitem(
        MODEL_PROFILES,
        MODEL,
        ModelProfile(
            omit_temperature=True, extra_body={"reasoning": {"effort": "none"}}
        ),
    )
    provider = Capture()
    client = LLMClient(providers=[provider])
    case = {
        "case_id": "offline",
        "source_text": "Use BLUE.",
        "candidates": [{"candidate_id": "cand_001", "canonical_text": "Use BLUE."}],
    }
    baseline = await original.evaluate(client, case, arm)
    before = provider.requests[:]
    provider.requests.clear()
    added = await original.evaluate(ModelOnlyClient(client), case, arm)
    after = provider.requests
    assert len(before) == len(after) == (4 if arm == "current_evidence" else 1)
    for old, new in zip(before, after, strict=True):
        assert old.pop("model") == "openai/gpt-5.6-luna"
        assert new.pop("model") == "openai/gpt-6-luna"
        assert old == new
        assert new["extra_body"]["reasoning"]["effort"] == "none"
        assert "temperature" not in new
        if arm == "luna_reference":
            assert old["max_tokens"] == 8192
    assert baseline == added
    await client.aclose()
