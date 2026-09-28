"""Focused checks for single-answer memory classification cards."""

from __future__ import annotations

import asyncio
import html
from types import SimpleNamespace

import pytest

from atagia.models.schemas_decisions import ChoiceAnswer, ScoreAnswer
from atagia.memory.extraction_cards import (
    CandidateDraft,
    build_classification_choice_question,
    extract_lean_with_cards,
    parse_classification_output,
    run_classification_card,
)
from atagia.models.schemas_memory import (
    ExtractionContextMessage,
    ExtractionConversationContext,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMError,
    LLMProvider,
)


class ChoiceClientMixin:
    async def complete_choice_questions(self, **kwargs):
        return await LLMClient.complete_choice_questions(self, **kwargs)

    async def complete_score_questions(self, **kwargs):
        return await LLMClient.complete_score_questions(self, **kwargs)


class ScriptedChoiceProvider(LLMProvider):
    name = "typesafe"
    supports_choices = True

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            choice_answers={
                question_id: ChoiceAnswer(
                    type="choice",
                    choice=self.answers[question_id],
                    probabilities={self.answers[question_id]: 1.0},
                    confidence=1.0,
                )
                for question_id in request.choice_questions or {}
            },
        )


def _context() -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        assistant_mode_id="general_qa",
        recent_messages=[],
        privacy_enforcement="off",
    )


@pytest.mark.asyncio
async def test_classification_sends_three_independent_answers_per_candidate() -> None:
    responses = {
        ("memory_extraction_kind_card", "cand_001"): "state_update",
        ("memory_extraction_kind_card", "cand_002"): "evidence",
        ("memory_extraction_scope_card", "cand_001"): "user",
        ("memory_extraction_scope_card", "cand_002"): "chat",
        ("memory_extraction_confidence_card", "cand_001"): "0.91",
        ("memory_extraction_confidence_card", "cand_002"): "0.42",
    }

    class Client(ChoiceClientMixin):
        def __init__(self) -> None:
            self.requests = []

        async def complete(self, request):
            self.requests.append(request)
            key = (
                request.metadata["purpose"],
                request.metadata.get("stage", request.metadata.get("memory_candidate_id")),
            )
            return SimpleNamespace(output_text=responses[key])

    client = Client()
    candidates = (
        CandidateDraft("cand_001", "The user is in Paris this week."),
        CandidateDraft("cand_002", "This chat uses branch sky-meadow."),
    )
    results = await asyncio.gather(
        *(
            run_classification_card(
                client,
                model="openrouter/openai/gpt-6-luna",
                card_name=card_name,
                message_text="I am in Paris this week; this chat uses branch sky-meadow.",
                role="user",
                context=_context(),
                allowed_write_scopes=("chat", "user"),
                occurred_at="2026-09-26T12:00:00+00:00",
                prior_chunk_context=None,
                candidates=candidates,
                metadata={},
                semaphore=asyncio.Semaphore(2),
            )
            for card_name in ("memory_kind", "memory_scope", "memory_confidence")
        )
    )

    assert len(client.requests) == 6
    assert {request.metadata["purpose"] for request in client.requests} == {
        "memory_extraction_kind_card",
        "memory_extraction_scope_card",
        "memory_extraction_confidence_card",
    }
    for request in client.requests:
        assert [message.role for message in request.messages] == ["system", "user"]
        assert request.messages[0].content == (
            "Decide one property of one memory candidate. "
            "Write only the requested answer. No JSON. No explanation."
        )
        assert request.choice_questions is None
        assert request.score_questions is None
        prompt = request.messages[1].content
        assert "<candidate_text>" in prompt
        assert "cand_001" not in prompt and "cand_002" not in prompt
        assert "<source_message role=\"user\">" in prompt
        assert request.response_schema is None
    assert [result.parsed for result in results] == [
        {"cand_001": "state_update", "cand_002": "evidence"},
        {"cand_001": "user", "cand_002": "chat"},
        {"cand_001": 0.91, "cand_002": 0.42},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "card_name", "model", "answer", "role", "source", "candidate_text",
        "scopes", "prior_context", "include_examples", "task_instruction",
    ),
    [
        (
            "memory_kind", "openrouter/openai/gpt-6-luna", "evidence", "user",
            "Nora: I prefer the blue notebook.", "Nora prefers the <blue> notebook.",
            ("chat", "user"), None, True, "Choose the memory type for this candidate.",
        ),
        (
            "memory_scope", "openai/gpt-4.1", "chat", "assistant",
            "My draft stays in this chat only.", "The draft stays in this chat.",
            ("user", "chat"), "An earlier draft was shared privately.", False,
            "Choose where this candidate should be stored.",
        ),
        (
            "memory_confidence", "openrouter/openai/gpt-6-luna", "0.68", "user",
            "The release may be delayed.", "The release may be delayed.",
            ("character", "user"), "The team discussed a tentative date.", True,
            "Answer with exactly one finite number between 0 and 1, inclusive.",
        ),
    ],
)
async def test_llm_classification_keeps_single_answer_payload(
    card_name: str,
    model: str,
    answer: str,
    role: str,
    source: str,
    candidate_text: str,
    scopes: tuple[str, ...],
    prior_context: str | None,
    include_examples: bool,
    task_instruction: str,
) -> None:
    class Client:
        def __init__(self) -> None:
            self.requests: list[LLMCompletionRequest] = []

        async def complete(self, request: LLMCompletionRequest) -> SimpleNamespace:
            self.requests.append(request)
            return SimpleNamespace(output_text=answer)

    client = Client()
    context = _context().model_copy(update={
        "recent_messages": [
            ExtractionContextMessage(role="user", content="Is the draft still private?", seq=2)
        ],
    })
    result = await run_classification_card(
        client,
        model=model,
        card_name=card_name,
        message_text=source,
        role=role,
        context=context,
        allowed_write_scopes=scopes,
        occurred_at="2026-09-26T12:00:00+00:00",
        prior_chunk_context=prior_context,
        candidates=(CandidateDraft("cand_007", candidate_text),),
        metadata={"trace_id": "classification-payload"},
        semaphore=asyncio.Semaphore(2),
        include_examples=include_examples,
    )

    assert len(client.requests) == 1
    request = client.requests[0]
    assert request.model == model
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].content == (
        "Decide one property of one memory candidate. "
        "Write only the requested answer. No JSON. No explanation."
    )
    prompt = request.messages[1].content
    assert prompt.index(task_instruction) < prompt.index("<source_message") < prompt.index("<candidate_text>")
    assert f'<source_message role="{role}">' in prompt
    assert html.escape(source) in prompt
    assert html.escape(candidate_text) in prompt
    assert prompt.count("<candidate_text>") == 1
    assert "Is the draft still private?" in prompt
    assert (prior_context or "(none)") in prompt
    assert ("Examples:" in prompt) is include_examples
    assert request.choice_questions is None
    assert request.score_questions is None
    assert request.response_schema is None
    assert request.max_output_tokens == 32
    assert request.metadata["purpose"] == f"memory_extraction_{card_name.removeprefix('memory_')}_card"
    assert request.metadata["memory_candidate_id"] == "cand_007"
    assert request.metadata["trace_id"] == "classification-payload"
    assert result.raw_output == answer
    assert result.parsed == {"cand_007": float(answer) if card_name == "memory_confidence" else answer}


@pytest.mark.asyncio
async def test_native_choices_batch_one_card_and_keep_each_candidate_target() -> None:
    source = "I did not authorize publication. Keep this chat's draft private."
    candidates = (
        CandidateDraft("cand_002", "The user did not authorize publication."),
        CandidateDraft("cand_001", "The user limited the draft to this chat."),
    )
    kind_provider = ScriptedChoiceProvider({"cand_002": "evidence", "cand_001": "contract_signal"})
    kind_client = LLMClient(providers=[kind_provider], structured_output_retry_attempts=0)
    kind = await run_classification_card(
        kind_client,
        model="typesafe/jev-1.13.0",
        card_name="memory_kind",
        message_text=source,
        role="user",
        context=_context(),
        allowed_write_scopes=("chat", "user"),
        occurred_at=None,
        prior_chunk_context="An earlier private draft was discussed.",
        candidates=candidates,
        metadata={},
        semaphore=asyncio.Semaphore(2),
    )
    assert kind.parsed == {"cand_002": "evidence", "cand_001": "contract_signal"}
    assert len(kind_provider.requests) == 1
    request = kind_provider.requests[0]
    assert list(request.choice_questions) == ["cand_002", "cand_001"]
    assert "I did not authorize publication." in request.messages[1].content
    assert request.messages[1].content.count("I did not authorize publication.") == 1
    assert "did not authorize" in request.choice_questions["cand_002"].instructions
    assert "earlier private draft" in request.messages[1].content
    assert all(
        "I did not authorize publication." not in question.instructions
        for question in request.choice_questions.values()
    )
    assert set(request.choice_questions["cand_002"].criteria) == {
        "evidence", "contract_signal", "state_update", "belief",
    }

    scope_provider = ScriptedChoiceProvider({"cand_002": "user", "cand_001": "chat"})
    scope_client = LLMClient(providers=[scope_provider], structured_output_retry_attempts=0)
    scope = await run_classification_card(
        scope_client,
        model="typesafe/jev-1.13.0",
        card_name="memory_scope",
        message_text=source,
        role="user",
        context=_context(),
        allowed_write_scopes=("chat", "user"),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=candidates,
        metadata={},
        semaphore=asyncio.Semaphore(2),
    )
    assert scope.parsed == {"cand_002": "user", "cand_001": "chat"}
    assert len(scope_provider.requests) == 1
    assert all(
        set(question.criteria) == {"chat", "user"}
        for question in scope_provider.requests[0].choice_questions.values()
    )


@pytest.mark.asyncio
async def test_native_scope_rejects_choice_outside_policy() -> None:
    provider = ScriptedChoiceProvider({"cand_001": "character"})
    client = LLMClient(providers=[provider], structured_output_retry_attempts=0)
    with pytest.raises(LLMError, match="unknown option"):
        await run_classification_card(
            client,
            model="typesafe/jev-1.13.0",
            card_name="memory_scope",
            message_text="I prefer brief replies.",
            role="user",
            context=_context(),
            allowed_write_scopes=("chat", "user"),
            occurred_at=None,
            prior_chunk_context=None,
            candidates=(CandidateDraft("cand_001", "The user prefers brief replies."),),
            metadata={},
            semaphore=asyncio.Semaphore(1),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    ["", "NaN", "inf", "-0.01", "1.01", "not-a-number", "0.5 0.6", "cand_001 0.5"],
)
async def test_confidence_rejects_missing_nonfinite_and_out_of_range(answer: str) -> None:
    class Client(ChoiceClientMixin):
        async def complete(self, request):
            return SimpleNamespace(output_text=answer)

    with pytest.raises(ValueError):
        await run_classification_card(
            Client(),
            model="openrouter/openai/gpt-6-luna",
            card_name="memory_confidence",
            message_text="I prefer short replies.",
            role="user",
            context=_context(),
            allowed_write_scopes=("chat", "user"),
            occurred_at=None,
            prior_chunk_context=None,
            candidates=(CandidateDraft("cand_001", "The user prefers short replies."),),
            metadata={},
            semaphore=asyncio.Semaphore(1),
        )


@pytest.mark.asyncio
async def test_memory_confidence_keeps_weighted_score_and_provider_certainty_separate() -> None:
    events = []

    class Recorder:
        @staticmethod
        def blob(value):
            return value

        def no_call(self, purpose, **kwargs):
            events.append((purpose, kwargs))

    class Client(ChoiceClientMixin):
        _diagnostic_recorder = Recorder()

        def __init__(self):
            self.requests = []

        async def complete(self, request):
            self.requests.append(request)
            answers = {}
            for candidate_id, question in request.score_questions.items():
                score = 2.6 if candidate_id == "cand_002" else 3.6
                probabilities = (
                    {"0": 0.0, "1": 0.0, "2": 0.4, "3": 0.6, "4": 0.0}
                    if candidate_id == "cand_002"
                    else {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.4, "4": 0.6}
                )
                answers[candidate_id] = ScoreAnswer(
                    type="score",
                    score=score,
                    legend={str(index): level for index, level in enumerate(question.criteria)},
                    probabilities=probabilities,
                    confidence=0.07,
                )
            return LLMCompletionResponse(
                provider="typesafe", model=request.model, score_answers=answers,
            )

    client = Client()
    candidates = (
        CandidateDraft("cand_002", "The user did not authorize publication."),
        CandidateDraft("cand_001", "The user expressed uncertainty about the date."),
    )
    result = await run_classification_card(
        client,
        model="typesafe/jev-1.13.0",
        card_name="memory_confidence",
        message_text="I did not authorize publication. I may have the date wrong.",
        role="user",
        context=_context(),
        allowed_write_scopes=("chat", "user"),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=candidates,
        metadata={"purpose": "spoofed", "user_id": "spoofed"},
        semaphore=asyncio.Semaphore(2),
    )

    assert result.parsed == {"cand_002": pytest.approx(0.65), "cand_001": pytest.approx(0.9)}
    assert result.raw_output == "2.6\n3.6"
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request.metadata["purpose"] == "memory_extraction_confidence_card"
    assert request.metadata["user_id"] == "usr_1"
    assert list(request.score_questions) == ["cand_002", "cand_001"]
    assert len(request.score_questions["cand_002"].criteria) == 5
    assert "did not authorize" in request.score_questions["cand_002"].instructions
    assert "source content" in request.score_questions["cand_002"].criteria[4]
    assert len(events) == 2
    assert events[0][1]["component"] == "extraction_confidence"
    assert events[0][1]["data"]["raw_score"] == 2.6
    assert events[0][1]["data"]["normalized_score"] == pytest.approx(0.65)
    assert events[0][1]["data"]["provider_confidence"] == 0.07


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("card_name", "answer"),
    [("memory_kind", "evidence"), ("memory_confidence", "0.65")],
)
async def test_classification_uses_shared_dispatch_capacity(
    card_name: str, answer: str,
) -> None:
    three_started = asyncio.Event()

    class Client(ChoiceClientMixin):
        def __init__(self):
            self.active = 0
            self.peak = 0

        async def complete(self, request):
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.active == 3:
                three_started.set()
            try:
                await three_started.wait()
                return SimpleNamespace(output_text=answer)
            finally:
                self.active -= 1

    client = Client()
    candidates = tuple(
        CandidateDraft(f"cand_{index:03d}", f"The user noted item {index}.")
        for index in range(1, 4)
    )
    result = await asyncio.wait_for(
        run_classification_card(
            client,
            model="openrouter/openai/gpt-6-luna",
            card_name=card_name,
            message_text="I noted three items.",
            role="user",
            context=_context(),
            allowed_write_scopes=("chat", "user"),
            occurred_at=None,
            prior_chunk_context=None,
            candidates=candidates,
            metadata={},
            semaphore=asyncio.Semaphore(3),
        ),
        timeout=2,
    )
    assert client.peak == 3
    assert set(result.parsed) == {candidate.candidate_id for candidate in candidates}


@pytest.mark.parametrize(
    ("card_name", "answer"),
    [
        ("memory_kind", "cand_001 evidence"),
        ("memory_kind", "none"),
        ("memory_scope", "user 0.9"),
        ("memory_scope", "workspace"),
    ],
)
def test_classification_rejects_rows_and_unknown_values(
    card_name: str, answer: str
) -> None:
    with pytest.raises(ValueError):
        parse_classification_output(card_name, answer)


@pytest.mark.asyncio
async def test_single_allowed_scope_reuses_policy_without_model_call() -> None:
    diagnostics = []

    class Recorder:
        def no_call(self, purpose, **kwargs):
            diagnostics.append((purpose, kwargs))

    class Client(ChoiceClientMixin):
        _diagnostic_recorder = Recorder()

        async def complete(self, request):
            raise AssertionError("Policy already decides the scope")

    result = await run_classification_card(
        Client(),
        model="openrouter/openai/gpt-6-luna",
        card_name="memory_scope",
        message_text="I prefer concise replies.",
        role="user",
        context=_context(),
        allowed_write_scopes=("chat",),
        occurred_at=None,
        prior_chunk_context=None,
        candidates=(CandidateDraft("cand_001", "The user prefers concise replies."),),
        metadata={},
        semaphore=asyncio.Semaphore(1),
    )

    assert result.parsed == {"cand_001": "chat"}
    assert diagnostics == [
        (
            "memory_extraction_scope_card",
            {
                "component": "extraction_scope",
                "user_id": "usr_1",
                "data": {
                    "card": "memory_scope",
                    "candidate_id": "cand_001",
                    "scope": "chat",
                    "reason": "policy_single_scope",
                },
            },
        )
    ]


@pytest.mark.asyncio
async def test_classification_cancels_other_candidates_on_failure() -> None:
    second_started = asyncio.Event()
    second_cancelled = asyncio.Event()

    class Client(ChoiceClientMixin):
        async def complete(self, request):
            if request.metadata["memory_candidate_id"] == "cand_001":
                await second_started.wait()
                raise ValueError("classification failed")
            second_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                second_cancelled.set()
                raise

    with pytest.raises(ValueError, match="classification failed"):
        await run_classification_card(
            Client(),
            model="openrouter/openai/gpt-6-luna",
            card_name="memory_kind",
            message_text="I prefer short replies and exact numbers.",
            role="user",
            context=_context(),
            allowed_write_scopes=("chat", "user"),
            occurred_at=None,
            prior_chunk_context=None,
            candidates=(
                CandidateDraft("cand_001", "The user prefers short replies."),
                CandidateDraft("cand_002", "The user prefers exact numbers."),
            ),
            metadata={},
            semaphore=asyncio.Semaphore(2),
        )
    assert second_cancelled.is_set()


@pytest.mark.asyncio
async def test_classification_cancels_other_cards_on_failure() -> None:
    started: set[str] = set()
    cancelled: set[str] = set()
    ready = asyncio.Event()

    class Client(ChoiceClientMixin):
        async def complete(self, request):
            purpose = request.metadata["purpose"]
            if purpose == "memory_extraction_candidate_card":
                return SimpleNamespace(output_text="cand_001 | The user prefers concise replies.")
            started.add(purpose)
            if {
                "memory_extraction_scope_card",
                "memory_extraction_confidence_card",
                "memory_extraction_index_card",
            } <= started:
                ready.set()
            if purpose == "memory_extraction_kind_card":
                await ready.wait()
                raise ValueError("kind failed")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.add(purpose)
                raise

    with pytest.raises(ValueError, match="kind failed"):
        await asyncio.wait_for(
            extract_lean_with_cards(
                llm_client=Client(),
                model="openrouter/openai/gpt-6-luna",
                evidence_model="openrouter/openai/gpt-6-luna",
                temporal_type_model="openrouter/openai/gpt-6-luna",
                date_model="openrouter/openai/gpt-6-luna,low",
                classification_models={
                    "memory_kind": "openrouter/openai/gpt-6-luna",
                    "memory_scope": "openrouter/openai/gpt-6-luna",
                    "memory_confidence": "openrouter/openai/gpt-6-luna",
                },
                message_text="I prefer concise replies.",
                role="user",
                context=_context(),
                resolved_policy=SimpleNamespace(preferred_memory_types=()),
                allowed_write_scopes=("chat", "user"),
                occurred_at=None,
                prior_chunk_context=None,
                metadata={},
                card_concurrency=8,
            ),
            timeout=5,
        )
    assert "memory_extraction_scope_card" in cancelled
    assert "memory_extraction_confidence_card" in cancelled
    assert cancelled == started - {"memory_extraction_kind_card"}


def test_scope_prompt_does_not_recommend_a_disallowed_destination() -> None:
    question = build_classification_choice_question(
        "memory_scope",
        candidate=CandidateDraft("cand_001", "The user prefers brief answers."),
        allowed_write_scopes=("user", "character"),
    )
    prompt = question.instructions
    assert set(question.criteria) == {"user", "character"}
    assert "allowed scope: user, character" in prompt
    assert "use chat" not in prompt
    assert "-> chat" not in prompt
