"""Focused offline coverage-member decision tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from atagia.diagnostics.recorder import DiagnosticRecorder
from atagia.memory.coverage_members_card import (
    IDENTITY_PURPOSE,
    IDENTITY_SYSTEM_PROMPT,
    MEMBERS_PURPOSE,
    MEMBERS_SYSTEM_PROMPT,
    build_identity_prompt,
    build_identity_questions,
    build_members_prompt,
    parse_identity_output,
    parse_members_output,
)
from atagia.memory.extraction_cards import (
    CandidateDraft,
    _source_context_block,
    run_coverage_members_card,
)
from atagia.models.schemas_decisions import ChoiceAnswer
from atagia.models.schemas_memory import (
    ExtractionContextMessage,
    ExtractionConversationContext,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMError,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMMessage,
    LLMProvider,
)


def _context() -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        assistant_mode_id="general_qa",
    )


class CoverageProvider(LLMProvider):
    name = "coverage-test"
    supports_choices = True

    def __init__(self, outputs: list[str | dict[str, str] | Exception]) -> None:
        self.outputs = iter(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        if request.choice_questions:
            assert isinstance(output, dict)
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                choice_answers={
                    question_id: ChoiceAnswer(
                        type="choice",
                        choice=choice,
                        probabilities={choice: 1.0},
                        confidence=1.0,
                    )
                    for question_id, choice in output.items()
                },
            )
        assert isinstance(output, str)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output,
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used")


def test_line_contract_preserves_punctuation_none_and_escaped_newline() -> None:
    assert parse_members_output('"ACME | East; R&D"\n"none"\n"María\\nSol"') == [
        "ACME | East; R&D",
        "none",
        "María\nSol",
    ]
    assert parse_members_output("none") == []
    assert parse_members_output('"A"\n') == ["A"]
    for malformed in ("", "[]", '"A"\nnone', '"A\nB"', '"Ana"\n"Ana"'):
        with pytest.raises(ValueError):
            parse_members_output(malformed)


def test_identity_requires_source_named_alias() -> None:
    assert parse_identity_output(
        'Lucía Ruiz',
        member="Dr. Ruiz",
        source_text="Mira sees Dr. Ruiz (Lucía Ruiz).",
    ) == "lucía ruiz"
    with pytest.raises(ValueError, match="absent from source"):
        parse_identity_output(
            'Dr. Unknown', member="Dr. Ruiz", source_text="Mira sees Dr. Ruiz."
        )


def test_identity_catalog_uses_only_candidate_members_and_stable_ids() -> None:
    labels = ["Lucía Ruiz", "Dr. Ruiz"]
    forward, forward_catalog = build_identity_questions(
        labels, candidate_text="Mira sees Dr. Ruiz (Lucía Ruiz)."
    )
    reverse, reverse_catalog = build_identity_questions(
        list(reversed(labels)), candidate_text="Mira sees Dr. Ruiz (Lucía Ruiz)."
    )

    assert forward == reverse
    assert forward_catalog == reverse_catalog == {
        "member_001": "Dr. Ruiz",
        "member_002": "Lucía Ruiz",
    }
    assert set(forward["member_001"].criteria) == {
        "self", "member_002", "not_listed"
    }
    assert "Lucía Ruiz" in forward["member_001"].criteria["member_002"]
    assert "Dr. Vale" not in str(forward)


@pytest.mark.asyncio
async def test_known_candidates_get_separate_membership_and_identity_decisions() -> None:
    provider = CoverageProvider(
        [
            '"Dr. Ruiz"\n"Dr. Okafor"',
            "none",
            "Lucía Ruiz",
            "Dr. Okafor",
        ]
    )
    client = LLMClient(provider_name=provider.name, providers=[provider])
    source = "Mira sees Dr. Ruiz (Lucía Ruiz) and Dr. Okafor. Dr. Vale was discussed."
    result = await run_coverage_members_card(
        client,
        model="openrouter/openai/test-model",
        message_text=source,
        role="user",
        context=_context(),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=(
            CandidateDraft("cand_001", "Mira sees Dr. Ruiz and Dr. Okafor."),
            CandidateDraft("cand_002", "Mira discussed Dr. Vale."),
        ),
        semaphore=asyncio.Semaphore(2),
    )

    assert [member.model_dump() for member in result.parsed["cand_001"]] == [
        {"member_key": "lucía ruiz", "display_text": "Dr. Ruiz"},
        {"member_key": "dr. okafor", "display_text": "Dr. Okafor"},
    ]
    assert result.parsed["cand_002"] == []
    assert [request.metadata["purpose"] for request in provider.requests] == [
        MEMBERS_PURPOSE,
        MEMBERS_PURPOSE,
        IDENTITY_PURPOSE,
        IDENTITY_PURPOSE,
    ]
    assert all(not request.finite_choice for request in provider.requests)
    assert all(request.choice_questions is None for request in provider.requests)
    assert all("cand_001" not in request.messages[1].content for request in provider.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "candidate_text", "labels", "identities"),
    [
        (
            "Nora consults Dr. V. (Valeria Montes) and Dr. Rahman.",
            "Nora consults Dr. V. and Dr. Rahman.",
            ["Dr. V.", "Dr. Rahman"],
            ["Valeria Montes", "Dr. Rahman"],
        ),
        (
            "Iris uses North Pier Credit (NPC) and South Harbor Bank.",
            "Iris uses NPC and South Harbor Bank.",
            ["NPC", "South Harbor Bank"],
            ["North Pier Credit", "South Harbor Bank"],
        ),
        (
            "Pau asked whether Dr. Tejada is taking new patients.",
            "Pau asked about Dr. Tejada.",
            [],
            [],
        ),
    ],
)
async def test_llm_requests_keep_baseline_member_identity_payload(
    source: str,
    candidate_text: str,
    labels: list[str],
    identities: list[str],
) -> None:
    provider = CoverageProvider([
        "\n".join(json.dumps(label) for label in labels) if labels else "none",
        *identities,
    ])
    client = LLMClient(provider_name=provider.name, providers=[provider])
    context = _context()
    model = "openrouter/openai/test-model"
    metadata = {"synthetic_case": "member_prompt_equivalence"}
    result = await run_coverage_members_card(
        client,
        model=model,
        message_text=source,
        role="user",
        context=context,
        occurred_at=None,
        prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", candidate_text),),
        metadata=metadata,
    )

    source_context = _source_context_block(
        message_text=source,
        role="user",
        context=context,
        occurred_at=None,
        prior_chunk_context=None,
    )
    expected_metadata = {
        "user_id": context.user_id,
        "conversation_id": context.conversation_id,
        "assistant_mode_id": context.assistant_mode_id,
        "purpose": IDENTITY_PURPOSE,
        **metadata,
    }
    list_request, *identity_requests = provider.requests
    assert list_request.model == model
    assert list_request.messages == [
        LLMMessage(role="system", content=MEMBERS_SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=build_members_prompt(
                candidate_text=candidate_text,
                source_context=source_context,
            ),
        ),
    ]
    assert len(identity_requests) == len(labels)
    for label, request in zip(labels, identity_requests, strict=True):
        assert request.model == model
        assert request.messages == [
            LLMMessage(role="system", content=IDENTITY_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=build_identity_prompt(
                    member=label,
                    candidate_text=candidate_text,
                    source_context=source_context,
                ),
            ),
        ]
        # The ordinary LLM client applies its output-token floor to the card's 256.
        assert request.max_output_tokens == 8192
        assert request.metadata == expected_metadata
        assert not request.finite_choice
        assert request.choice_questions is None
    assert [member.member_key for member in result.parsed["cand_001"]] == [
        identity.casefold() for identity in identities
    ]


@pytest.mark.asyncio
async def test_llm_identity_generation_receives_recent_and_chunk_context() -> None:
    provider = CoverageProvider(['"Dr. V."', "Valeria Montes"])
    client = LLMClient(provider_name=provider.name, providers=[provider])
    context = ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_2",
        assistant_mode_id="general_qa",
        recent_messages=[
            ExtractionContextMessage(
                id="msg_1",
                role="user",
                content="Dr. V. is Valeria Montes.",
                seq=1,
            )
        ],
    )
    result = await run_coverage_members_card(
        client,
        model="openrouter/openai/test-model",
        message_text="Nora consults Dr. V.",
        role="user",
        context=context,
        occurred_at="2026-09-01T10:00:00+00:00",
        prior_chunk_context="Earlier source chunk about Nora's appointments.",
        candidates=(CandidateDraft("cand_001", "Nora consults Dr. V."),),
    )

    assert result.parsed["cand_001"][0].member_key == "valeria montes"
    identity_prompt = provider.requests[1].messages[1].content
    assert "Valeria Montes" in identity_prompt
    assert "Earlier source chunk about Nora" in identity_prompt
    assert "2026-09-01T10:00:00+00:00" in identity_prompt
    assert provider.requests[1].messages[0].content == IDENTITY_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_native_choice_batches_aliases_and_uses_source_named_identity() -> None:
    provider = CoverageProvider([
        '"Dr. Ruiz"\n"Lucía Ruiz"',
        {"member_002": "self", "member_001": "member_002"},
    ])
    client = LLMClient(provider_name=provider.name, providers=[provider])
    result = await run_coverage_members_card(
        client,
        model="openrouter/openai/test-generator",
        identity_model="typesafe/jev-1.13.0",
        message_text="Mira sees Dr. Ruiz, also named Lucía Ruiz.",
        role="user",
        context=_context(),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", "Mira sees Dr. Ruiz and Lucía Ruiz."),),
    )

    assert [member.model_dump() for member in result.parsed["cand_001"]] == [
        {"member_key": "lucía ruiz", "display_text": "Dr. Ruiz"}
    ]
    assert len(provider.requests) == 2
    assert provider.requests[0].model == "openrouter/openai/test-generator"
    assert provider.requests[1].model == "typesafe/jev-1.13.0"
    assert set(provider.requests[1].choice_questions) == {"member_001", "member_002"}


@pytest.mark.asyncio
async def test_distinct_homonyms_remain_distinct_without_established_alias() -> None:
    provider = CoverageProvider([
        '"Ana García (North Clinic)"\n"Ana García (South Clinic)"',
        {"member_002": "self", "member_001": "self"},
    ])
    client = LLMClient(provider_name=provider.name, providers=[provider])
    result = await run_coverage_members_card(
        client,
        model="openrouter/openai/test-generator",
        identity_model="typesafe/jev-1.13.0",
        message_text="Mira sees Ana García (North Clinic) and Ana García (South Clinic).",
        role="user",
        context=_context(),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", "Mira sees two doctors named Ana García."),),
    )

    assert [member.member_key for member in result.parsed["cand_001"]] == [
        "ana garcía (north clinic)", "ana garcía (south clinic)"
    ]
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_not_listed_uses_generator_and_invalid_choice_does_not() -> None:
    source = "Mira sees Dr. Ruiz (Lucía Ruiz) and Dr. Okafor."
    candidates = (CandidateDraft("cand_001", "Mira sees Dr. Ruiz and Dr. Okafor."),)
    provider = CoverageProvider([
        '"Dr. Ruiz"\n"Dr. Okafor"',
        {"member_002": "not_listed", "member_001": "self"},
        "Lucía Ruiz",
    ])
    client = LLMClient(provider_name=provider.name, providers=[provider])
    result = await asyncio.wait_for(
        run_coverage_members_card(
            client,
            model="openrouter/openai/test-generator",
            identity_model="typesafe/jev-1.13.0",
            message_text=source,
            role="user",
            context=_context(),
            occurred_at=None,
            prior_chunk_context=None,
            candidates=candidates,
            semaphore=asyncio.Semaphore(1),
        ),
        timeout=2,
    )
    assert [member.member_key for member in result.parsed["cand_001"]] == [
        "lucía ruiz", "dr. okafor"
    ]
    assert [request.model for request in provider.requests] == [
        "openrouter/openai/test-generator",
        "typesafe/jev-1.13.0",
        "openrouter/openai/test-generator",
    ]

    invalid = CoverageProvider([
        '"Dr. Ruiz"\n"Dr. Okafor"',
        {"member_001": "self", "member_002": "foreign_id"},
    ])
    invalid_client = LLMClient(provider_name=invalid.name, providers=[invalid])
    with pytest.raises(LLMError, match="unknown option"):
        await run_coverage_members_card(
            invalid_client,
            model="openrouter/openai/test-generator",
            identity_model="typesafe/jev-1.13.0",
            message_text=source,
            role="user",
            context=_context(),
            occurred_at=None,
            prior_chunk_context=None,
            candidates=candidates,
        )
    assert len(invalid.requests) == 2


@pytest.mark.asyncio
async def test_native_override_keeps_generation_when_no_identity_catalog_exists() -> None:
    provider = CoverageProvider(['"Dr. Ruiz"', "Lucía Ruiz"])
    client = LLMClient(provider_name=provider.name, providers=[provider])
    result = await run_coverage_members_card(
        client,
        model="openrouter/openai/test-generator",
        identity_model="typesafe/jev-1.13.0",
        message_text="Mira sees Dr. Ruiz (Lucía Ruiz).",
        role="user",
        context=_context(),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", "Mira sees Dr. Ruiz."),),
    )

    assert result.parsed["cand_001"][0].member_key == "lucía ruiz"
    assert [request.model for request in provider.requests] == [
        "openrouter/openai/test-generator",
        "openrouter/openai/test-generator",
    ]
    assert all(request.choice_questions is None for request in provider.requests)


@pytest.mark.asyncio
async def test_llm_identity_override_uses_generation_even_with_catalog() -> None:
    provider = CoverageProvider([
        '"Dr. Ruiz"\n"Dr. Okafor"', "Lucía Ruiz", "Dr. Okafor"
    ])
    client = LLMClient(provider_name=provider.name, providers=[provider])
    result = await run_coverage_members_card(
        client,
        model="openrouter/openai/test-generator",
        identity_model="openrouter/openai/test-identity",
        message_text="Mira sees Dr. Ruiz (Lucía Ruiz) and Dr. Okafor.",
        role="user",
        context=_context(),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", "Mira sees Dr. Ruiz and Dr. Okafor."),),
    )

    assert [member.member_key for member in result.parsed["cand_001"]] == [
        "lucía ruiz", "dr. okafor"
    ]
    assert [request.model for request in provider.requests] == [
        "openrouter/openai/test-generator",
        "openrouter/openai/test-identity",
        "openrouter/openai/test-identity",
    ]
    assert all(not request.finite_choice for request in provider.requests)


@pytest.mark.asyncio
async def test_large_member_list_does_not_build_quadratic_choice_catalog() -> None:
    labels = [f"Member {index}" for index in range(17)]
    provider = CoverageProvider([
        "\n".join(json.dumps(label) for label in labels),
        *labels,
    ])
    client = LLMClient(provider_name=provider.name, providers=[provider])
    result = await run_coverage_members_card(
        client,
        model="openrouter/openai/test-generator",
        identity_model="typesafe/jev-1.13.0",
        message_text="The team includes " + ", ".join(labels) + ".",
        role="user",
        context=_context(),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", "The team has 17 members."),),
    )

    assert len(result.parsed["cand_001"]) == 17
    assert len(provider.requests) == 18
    assert all(request.choice_questions is None for request in provider.requests)


@pytest.mark.asyncio
async def test_identity_catalogs_stay_with_their_own_candidate() -> None:
    class PartitionProvider(LLMProvider):
        name = "coverage-partition-test"
        supports_choices = True

        def __init__(self) -> None:
            self.requests: list[LLMCompletionRequest] = []

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            self.requests.append(request)
            if request.choice_questions:
                return LLMCompletionResponse(
                    provider=self.name,
                    model=request.model,
                    choice_answers={
                        question_id: ChoiceAnswer(
                            type="choice",
                            choice="self",
                            probabilities={"self": 1.0},
                            confidence=1.0,
                        )
                        for question_id in request.choice_questions
                    },
                )
            prompt = request.messages[-1].content
            output = (
                '"Ana García (North Clinic)"\n"Ben"'
                if "<candidate>\nMira sees Ana García (North Clinic)" in prompt
                else '"Ana García (South Clinic)"\n"Cara"'
            )
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text=output
            )

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Embeddings are not used")

    provider = PartitionProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])
    result = await run_coverage_members_card(
        client,
        model="openrouter/openai/test-generator",
        identity_model="typesafe/jev-1.13.0",
        message_text=(
            "Mira sees Ana García (North Clinic) and Ben. "
            "Mira also sees Ana García (South Clinic) and Cara."
        ),
        role="user",
        context=_context(),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=(
            CandidateDraft("cand_001", "Mira sees Ana García (North Clinic) and Ben."),
            CandidateDraft("cand_002", "Mira sees Ana García (South Clinic) and Cara."),
        ),
    )

    assert set(result.parsed) == {"cand_001", "cand_002"}
    choices = [request for request in provider.requests if request.choice_questions]
    assert len(choices) == 2
    catalogs = [str(request.choice_questions) for request in choices]
    assert any("North Clinic" in item and "South Clinic" not in item for item in catalogs)
    assert any("South Clinic" in item and "North Clinic" not in item for item in catalogs)
    assert len(provider.requests) == 4


@pytest.mark.asyncio
async def test_llm_identity_generations_use_available_shared_concurrency() -> None:
    class ConcurrentProvider(LLMProvider):
        name = "coverage-concurrency-test"

        def __init__(self) -> None:
            self.active = 0
            self.peak = 0
            self.all_started = asyncio.Event()

        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            if request.metadata["purpose"] == MEMBERS_PURPOSE:
                output = '"Alice"\n"Bob"\n"Carol"'
            else:
                assert request.finite_choice is False
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.active == 3:
                    self.all_started.set()
                try:
                    await asyncio.wait_for(self.all_started.wait(), timeout=2)
                finally:
                    self.active -= 1
                output = request.messages[-1].content.split("<member>\n", 1)[1].split(
                    "\n</member>", 1
                )[0]
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text=output
            )

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Embeddings are not used")

    provider = ConcurrentProvider()
    client = LLMClient(provider_name=provider.name, providers=[provider])
    result = await asyncio.wait_for(
        run_coverage_members_card(
            client,
            model="openrouter/openai/test-model",
            message_text="The team includes Alice, Bob, and Carol.",
            role="user",
            context=_context(),
            occurred_at=None,
            prior_chunk_context=None,
            candidates=(CandidateDraft("cand_001", "The team includes three people."),),
            semaphore=asyncio.Semaphore(3),
        ),
        timeout=3,
    )
    assert provider.peak == 3
    assert len(result.parsed["cand_001"]) == 3


class CancellingProvider(LLMProvider):
    name = "coverage-cancellation-test"

    def __init__(self, failing_purpose: str) -> None:
        self.failing_purpose = failing_purpose
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.calls: list[str] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata["purpose"])
        self.calls.append(purpose)
        if purpose == MEMBERS_PURPOSE and self.failing_purpose == IDENTITY_PURPOSE:
            output = '"Dr. A"\n"Dr. B"'
        elif purpose == self.failing_purpose:
            if self.calls.count(purpose) == 1:
                await self.started.wait()
                output = "malformed"
            else:
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
        else:
            raise AssertionError(f"Unexpected purpose: {purpose}")
        return LLMCompletionResponse(
            provider=self.name, model=request.model, output_text=output
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used")


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_purpose", [MEMBERS_PURPOSE, IDENTITY_PURPOSE])
async def test_failing_sibling_cancels_other_coverage_calls(
    failing_purpose: str, tmp_path: Path
) -> None:
    provider = CancellingProvider(failing_purpose)
    recorder = DiagnosticRecorder(tmp_path)
    client = LLMClient(
        provider_name=provider.name,
        providers=[provider],
        diagnostic_recorder=recorder,
    )
    candidates = (
        CandidateDraft("cand_001", "Mira sees Dr. A and Dr. B."),
        CandidateDraft("cand_002", "Mira sees Dr. B."),
    ) if failing_purpose == MEMBERS_PURPOSE else (
        CandidateDraft("cand_001", "Mira sees Dr. A and Dr. B."),
    )
    expected_error = (
        "requires JSON strings" if failing_purpose == MEMBERS_PURPOSE
        else "absent from source"
    )
    with pytest.raises(ValueError, match=expected_error):
        await asyncio.wait_for(
            run_coverage_members_card(
                client,
                model="openrouter/openai/test-model",
                message_text="Mira sees Dr. A and Dr. B.",
                role="user",
                context=_context(),
                occurred_at=None,
                prior_chunk_context=None,
                candidates=candidates,
                semaphore=asyncio.Semaphore(2),
            ),
            timeout=2,
        )
    assert provider.cancelled.is_set()
    assert provider.calls.count(failing_purpose) == 2
    recorder.close()
    events = [
        json.loads(line) for line in (recorder.root / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        event["kind"] == "provider_attempt"
        and event.get("phase") == "end"
        and event["status"] == "cancelled"
        and event["purpose"] == failing_purpose
        for event in events
    )
