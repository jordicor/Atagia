"""Focused checks for single-candidate belief card execution."""

from __future__ import annotations

import asyncio

import pytest

from atagia.memory.extraction_cards import CandidateDraft, run_belief_cards
from atagia.models.schemas_memory import ExtractionConversationContext
from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse


class BlockingBeliefClient:
    def __init__(self) -> None:
        self.second_started = asyncio.Event()
        self.second_cancelled = asyncio.Event()
        self.purposes: list[str] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata["purpose"])
        self.purposes.append(purpose)
        if purpose != "memory_extraction_belief_key_card":
            raise AssertionError(f"Unexpected belief value call: {purpose}")
        candidate_text = request.messages[-1].content.split("<candidate>\n", 1)[1].split(
            "\n</candidate>", 1
        )[0]
        if candidate_text == "The user avoids automatic edits.":
            await self.second_started.wait()
            return LLMCompletionResponse(
                provider="test", model=request.model, output_text="invalid.key!"
            )
        if candidate_text != "The user prefers concise reviews.":
            raise AssertionError(f"Unexpected candidate: {candidate_text}")
        self.second_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.second_cancelled.set()
            raise
        raise AssertionError("The waiting request should be cancelled")


@pytest.mark.asyncio
async def test_invalid_belief_key_cancels_sibling_before_returning() -> None:
    client = BlockingBeliefClient()
    context = ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id=None,
        assistant_mode_id="coding_debug",
        recent_messages=[],
    )
    with pytest.raises(ValueError, match="claim_key"):
        await asyncio.wait_for(
            run_belief_cards(
                client,
                model="openrouter/openai/gpt-6-luna",
                message_text="Avoid automatic edits and keep reviews concise.",
                role="user",
                context=context,
                occurred_at=None,
                prior_chunk_context=None,
                candidates=(
                    CandidateDraft("cand_001", "The user avoids automatic edits."),
                    CandidateDraft("cand_002", "The user prefers concise reviews."),
                ),
                metadata={},
                semaphore=asyncio.Semaphore(2),
                include_examples=False,
            ),
            timeout=2,
        )
    assert client.second_cancelled.is_set()
    assert client.purposes == [
        "memory_extraction_belief_key_card",
        "memory_extraction_belief_key_card",
    ]
