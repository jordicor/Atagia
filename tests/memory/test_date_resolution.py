"""Promoted date operations, durable annotations, and ordinary-client routing."""

from hashlib import sha256

import pytest

from atagia.memory.date_resolution import (
    CLEAN,
    build_date_resolution_prompt,
    parse_date_resolution,
    pending_date_analysis,
    read_persisted_date_resolution,
    resolve_date,
)
from atagia.services.llm_client import LLMClient, LLMCompletionResponse, LLMProvider
from atagia.services.providers.openrouter import OpenRouterProvider


@pytest.mark.parametrize("answer,expected", [
    ("exact|2030-12-30|days|+3", "2031-01-02"),
    ("exact|2036-03-05|weeks|-2", "2036-02-20"),
    ("exact|2037-01-31|months|1", "2037-02-28"),
    ("exact|2040-02-29|years|1", "2041-02-28"),
    ("uncertain|2034-03-31|months|-1.5", "2034-02-13"),
    ("uncertain|2035-12-22|months|2.5", "2036-03-08"),
    ("uncertain|2031-08-18|months|-0.5", "2031-08-03"),
    ("exact|2030-05-10|years|-1.5", "2028-11-10"),
    ("exact|2034-11-23|days|0", "2034-11-23"),
    ("exact|ref|next_weekday|THURSDAY", "2030-07-18"),
    ("exact|ref|previous_weekday|wednesday", "2030-07-10"),
    ("exact|ref|next_weekday|saturday", "2030-07-13"),
    ("unknown", None),
])
def test_selected_calendar_and_estimate_policy(answer, expected):
    result = parse_date_resolution(answer, "2030-07-11")
    assert result.resolved_date == expected
    assert result.status == "completed"
    assert result.certainty == answer.split("|")[0]


@pytest.mark.parametrize("answer", [
    "days|3", "exact|unknown|days|0", "exact|20300510|days|3",
    "exact|2030-05-10|days|NaN", "exact|2030-05-10|hours|24",
    "exact|2030-05-10|weeks|0.5|days|1", "exact|2030-02-30|days|0",
    "unknown|0000-00-00|days|0", "exact|ref|next_weekday|invalid",
])
def test_invalid_outputs_remain_visible_failures(answer):
    with pytest.raises(ValueError):
        parse_date_resolution(answer, "2030-05-10")


def test_selected_prompt_fidelity_and_source_calendar_day():
    # SHA-256 of the approved CLEAN instructions in the sealed experiment.
    assert sha256(CLEAN.encode()).hexdigest() == (
        "1e6fba027589c72e623c4608638c2f9f1eee62e17ee8798fc0d02e583b2d741e"
    )
    prompt = build_date_resolution_prompt(
        "Demà tinc la visita al taller.", "2026-04-30T23:30:00-05:00",
    )
    assert prompt == (
        "Reference date: 2026-04-30\nText: Demà tinc la visita al taller.\n\n"
        + CLEAN + "\n\nReturn only the result."
    )


class DateProvider(LLMProvider):
    name = "openrouter"

    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name, model=request.model, output_text=self.answer,
        )


@pytest.mark.asyncio
async def test_normal_client_keeps_low_routing_floor_and_annotation_reuse():
    provider = DateProvider("exact|ref|days|+8")
    client = LLMClient(providers=[provider])
    text = "The meeting is in eight days."
    annotation = await resolve_date(
        client, "openrouter/openai/gpt-6-luna,low", text,
        "2030-05-10T23:30:00-05:00", {"user_id": "synthetic-user"},
    )
    assert annotation.resolved_date == "2030-05-18"
    assert annotation.certainty == "exact"
    request, = provider.requests
    assert request.model == "openai/gpt-6-luna"
    assert request.metadata["provider_extra_body"]["reasoning"]["effort"] == "low"
    assert request.metadata["purpose"] == "memory_date_resolution"
    assert request.metadata["user_id"] == "synthetic-user"
    assert request.max_output_tokens == 8192
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.messages[0].content == build_date_resolution_prompt(text, "2030-05-10")
    adapter = OpenRouterProvider(
        api_key="stub", site_url="https://example.test", app_name="test", client=object(),
    )
    wire = adapter._completion_kwargs(request, stream=False)
    assert wire["model"] == "openai/gpt-6-luna"
    assert wire["messages"] == [{"role": "user", "content": request.messages[0].content}]
    assert wire["extra_body"]["reasoning"]["effort"] == "low"
    assert wire["extra_body"]["provider"]["allow_fallbacks"] is False
    assert wire["max_tokens"] == 8192
    assert "temperature" not in wire
    assert "response_format" not in wire
    payload = {"date_resolution": annotation.model_dump(mode="json")}
    assert read_persisted_date_resolution(payload, text, "2030-05-10") == annotation
    assert read_persisted_date_resolution(payload, text + " ", "2030-05-10") is None
    assert read_persisted_date_resolution(payload, text, "2030-05-11") is None
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,status,certainty", [
    ("unknown", "completed", "unknown"),
    ("analyze", "pending_analysis", None),
])
async def test_unknown_and_analysis_remain_distinct_and_durable(answer, status, certainty):
    provider = DateProvider(answer)
    annotation = await resolve_date(
        LLMClient(providers=[provider]), "openrouter/openai/gpt-6-luna,low",
        "The visit was several months ago.", "2030-05-10",
    )
    assert annotation.status == status
    assert annotation.certainty == certainty
    assert annotation.resolved_date is None
    payload = {"date_resolution": annotation.model_dump(mode="json")}
    assert read_persisted_date_resolution(
        payload, "The visit was several months ago.", "2030-05-10",
    ) == annotation
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_malformed_completion_is_not_repaired_or_retried():
    provider = DateProvider("exact|unknown|days|0")
    with pytest.raises(ValueError):
        await resolve_date(
            LLMClient(providers=[provider]), "openrouter/openai/gpt-6-luna,low",
            "The meeting is in eight days.", "2030-05-10",
        )
    assert len(provider.requests) == 1


def test_missing_source_reference_remains_pending_and_invalid_annotations_are_not_reused():
    text = "The meeting is in eight days."
    annotation = pending_date_analysis(text)
    assert annotation.reference_date is None
    assert annotation.status == "pending_analysis"
    payload = {"date_resolution": annotation.model_dump(mode="json")}
    assert read_persisted_date_resolution(payload, text, None) == annotation
    assert read_persisted_date_resolution(payload, text, "2030-05-10") is None
    payload["date_resolution"]["status"] = "completed"
    assert read_persisted_date_resolution(payload, text, None) is None
    assert read_persisted_date_resolution({}, text, "2030-05-10") is None
