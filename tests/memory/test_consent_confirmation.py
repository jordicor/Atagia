"""Tests for consent confirmation helpers."""

from __future__ import annotations

import pytest

from atagia.memory.consent_confirmation import (
    ConsentResponseIntent,
    category_plural_label,
    classify_confirmation_response,
    safe_confirmation_label,
)
from atagia.models.schemas_memory import MemoryCategory
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMCompletionResponse, LLMProvider


class StubStructuredProvider(LLMProvider):
    name = "stub-structured"

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.output_text,
        )


def _client(output_text: str) -> tuple[LLMClient[object], StubStructuredProvider]:
    provider = StubStructuredProvider(output_text)
    return LLMClient(provider_name=provider.name, providers=[provider]), provider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_text", "output_text", "expected"),
    [
        ("sí, guarda eso", '{"intent":"confirm"}', ConsentResponseIntent.CONFIRM),
        ("no, por favor", '{"intent":"deny"}', ConsentResponseIntent.DENY),
        ("maybe later", '{"intent":"ambiguous"}', ConsentResponseIntent.AMBIGUOUS),
    ],
)
async def test_classify_confirmation_response_uses_llm_intent(
    message_text: str,
    output_text: str,
    expected: ConsentResponseIntent,
) -> None:
    client, provider = _client(output_text)

    result = await classify_confirmation_response(
        message_text,
        client,
        prompt_text="Before I answer, I noted your phone number earlier. Want me to keep it for next time?",
    )

    assert result is expected
    assert provider.requests[0].metadata["purpose"] == "consent_confirmation_intent"
    assert "Want me to keep it for next time?" in provider.requests[0].messages[1].content
    assert message_text in provider.requests[0].messages[1].content


@pytest.mark.asyncio
async def test_classify_confirmation_response_supports_mixed_answers_via_schema() -> None:
    client, provider = _client('{"intent":"ambiguous"}')

    result = await classify_confirmation_response(
        "Yes for the phone, but no for the address.",
        client,
        prompt_text="Before I answer, I noted your phone number earlier. Want me to keep it for next time?",
    )

    assert result is ConsentResponseIntent.AMBIGUOUS
    assert "<assistant_confirmation_prompt>" in provider.requests[0].messages[1].content
    assert "Yes for the phone, but no for the address." in provider.requests[0].messages[1].content


def test_safe_confirmation_label_prefers_sanitized_index_text() -> None:
    assert (
        safe_confirmation_label("Banking card PIN 4512 for emergency access", MemoryCategory.PIN_OR_PASSWORD)
        == "Banking card PIN"
    )


def test_safe_confirmation_label_falls_back_to_category_label() -> None:
    assert safe_confirmation_label(None, MemoryCategory.FINANCIAL) == "your financial detail"
    assert category_plural_label(MemoryCategory.FINANCIAL) == "your financial details"


def test_safe_confirmation_label_trims_secret_suffix_after_password_keyword() -> None:
    assert (
        safe_confirmation_label("WiFi password opensesame", MemoryCategory.PIN_OR_PASSWORD)
        == "WiFi password"
    )
