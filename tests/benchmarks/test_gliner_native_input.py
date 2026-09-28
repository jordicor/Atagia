"""No-inference checks for the GLiNER-native prompt transport challenger."""

from __future__ import annotations

import pytest

from atagia.memory.card_prompt import compose_card_prompt
from atagia.memory.consequence_detector import _card_task_parts
from atagia.memory.need_detector import _card_task
from benchmarks.local_decision_cards.cases import SMOKE_CASES, capture_request
from benchmarks.local_decision_cards.gliner_native_input import orient_request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card,question_id",
    [
        ("context_reuse", "context_reuse"),
        ("need_detector_exact", "exact"),
        ("consequence_sentiment", "outcome_sentiment"),
    ],
)
async def test_moves_rules_into_native_instruction_without_losing_state(
    card: str, question_id: str
) -> None:
    case = next(item for item in SMOKE_CASES if item.card == card)
    original = await capture_request(case, "typesafe/jev-latest")
    original_dump = original.model_dump(mode="json")

    oriented = orient_request(original)

    assert original.model_dump(mode="json") == original_dump
    assert oriented.model == original.model
    assert oriented.metadata == original.metadata
    assert oriented.max_output_tokens == original.max_output_tokens
    assert len(oriented.messages) == 1
    assert oriented.messages[0].role == "user"
    assert set(oriented.choice_questions or {}) == {question_id}
    old_question = original.choice_questions[question_id]
    new_question = oriented.choice_questions[question_id]
    assert new_question.criteria == old_question.criteria
    assert new_question.type == old_question.type
    assert new_question.instructions.startswith(original.messages[0].content + "\n\n")
    assert new_question.instructions.endswith("\n\n" + old_question.instructions)
    assert case.message in oriented.messages[0].content or card != "context_reuse"

    old_prompt = original.messages[1].content
    if card == "context_reuse":
        intro, separator, state_tail = old_prompt.partition("\n\n<current_mode_id>")
        assert separator
        assert oriented.messages[0].content == "<current_mode_id>" + state_tail
        assert new_question.instructions == "\n\n".join(
            (original.messages[0].content, intro, old_question.instructions)
        )
        assert "<previous_user_message>" in oriented.messages[0].content
        assert "<new_user_message>" in oriented.messages[0].content
        return

    marker = "</atagia_process_metadata>\n\n"
    metadata, remainder = old_prompt.split(marker, maxsplit=1)
    metadata += "</atagia_process_metadata>"
    if card == "need_detector_exact":
        instruction, examples, _ = _card_task("exact")
        assert oriented.messages[0].content.startswith("Reference time: ")
        assert "\nRole: user\n" in oriented.messages[0].content
        assert "\nRecent messages:\n" in oriented.messages[0].content
        assert "\nSaved memory languages:\n" in oriented.messages[0].content
    else:
        instruction, examples = _card_task_parts("sentiment")
        assert oriented.messages[0].content.startswith('<source_message role="user">')
        assert "<user_message>" in oriented.messages[0].content
        assert "<assistant_history>" in oriented.messages[0].content

    rendered_tasks = [
        compose_card_prompt(instruction, examples, include_examples=include)
        for include in (True, False)
    ]
    task = next(task for task in rendered_tasks if remainder.endswith("\n\n" + task))
    assert remainder == oriented.messages[0].content + "\n\n" + task
    assert new_question.instructions == "\n\n".join(
        (original.messages[0].content, metadata, task, old_question.instructions)
    )
    assert "<atagia_process_metadata>" not in oriented.messages[0].content
    assert "Examples:" not in oriented.messages[0].content


@pytest.mark.asyncio
async def test_rejects_unrecognized_production_layout() -> None:
    case = next(item for item in SMOKE_CASES if item.card == "consequence_sentiment")
    original = await capture_request(case, "typesafe/jev-latest")
    malformed = original.model_copy(
        update={
            "messages": [
                original.messages[0],
                original.messages[1].model_copy(
                    update={"content": original.messages[1].content.replace(
                        "</atagia_process_metadata>", "</unknown_metadata>"
                    )}
                ),
            ]
        }
    )

    with pytest.raises(ValueError, match="process metadata"):
        orient_request(malformed)
