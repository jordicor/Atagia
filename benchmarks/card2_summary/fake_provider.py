"""In-process FAKE provider for the offline self-test (no network, no keys).

It returns a canned non-empty summary per range, optionally injects one
empty-then-valid response on a chosen range to exercise the engine's per-range
retry, and optionally sleeps to force concurrency overlap so the observed cap is
meaningful. It implements the same ``LLMProvider`` surface as a real provider, so
the harness pipeline runs end to end exactly as it would live.
"""

from __future__ import annotations

import asyncio

from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

_SUMMARY_PURPOSE = "summary_chunk_segmentation_summaries_card"


class FakeSummaryProvider(LLMProvider):
    """Canned, deterministic provider for offline harness validation."""

    name = "card2-fake"
    supports_embeddings = False

    def __init__(
        self,
        *,
        summary_delay_seconds: float = 0.0,
        retry_range: tuple[int, int] | None = None,
    ) -> None:
        # ``retry_range`` returns "" on its FIRST call (forcing one retry) then a
        # valid summary; every other range succeeds on the first call.
        self._summary_delay_seconds = summary_delay_seconds
        self._retry_range = retry_range
        self._retry_range_seen = False

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata.get("purpose"))
        if purpose != _SUMMARY_PURPOSE:
            # The harness only drives card 2; any other purpose is unexpected.
            raise AssertionError(
                f"FakeSummaryProvider received unexpected purpose {purpose!r}"
            )
        if self._summary_delay_seconds:
            await asyncio.sleep(self._summary_delay_seconds)
        range_start = int(request.metadata["range_start"])
        range_end = int(request.metadata["range_end"])
        if (
            self._retry_range is not None
            and (range_start, range_end) == self._retry_range
            and not self._retry_range_seen
        ):
            self._retry_range_seen = True
            output_text = ""  # empty -> engine retries this range once
        else:
            output_text = (
                f"Synthetic summary of messages {range_start}-{range_end}."
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output_text,
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("FakeSummaryProvider does not embed")
