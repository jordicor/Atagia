"""GLiNER-native input challenger for the frozen local decision benchmark.

This mechanical transport variant moves production instructions out of the
classified state and into the native choice instruction field. It changes no
decision rubric, example, authority metadata, option, or source-data text.
"""

from __future__ import annotations

from atagia.memory.card_prompt import compose_card_prompt
from atagia.memory.consequence_detector import _card_task_parts as _consequence_card_task_parts
from atagia.memory.need_detector import _card_task as _need_card_task
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage


_METADATA_END = "</atagia_process_metadata>"


def _split_context_reuse(prompt: str) -> tuple[str, str]:
    intro, separator, state = prompt.partition("\n\n<current_mode_id>")
    if not separator or not intro or not state.endswith("</new_user_message>"):
        raise ValueError("Context reuse prompt does not match the production builder")
    state = "<current_mode_id>" + state
    if (
        state.count("<current_mode_id>") != 1
        or state.count("<previous_user_message>") != 1
        or state.count("<new_user_message>") != 1
    ):
        raise ValueError("Context reuse state does not match the production builder")
    return intro, state


def _split_analytical_card(
    prompt: str, *, card: str
) -> tuple[str, str, str]:
    separator = _METADATA_END + "\n\n"
    if not prompt.startswith("<atagia_process_metadata>\n") or prompt.count(separator) != 1:
        raise ValueError("Card prompt has no unique production process metadata block")
    metadata, remainder = prompt.split(separator, maxsplit=1)
    metadata += _METADATA_END

    if card == "exact":
        instruction, examples, _ = _need_card_task("exact")
    elif card == "sentiment":
        instruction, examples = _consequence_card_task_parts("sentiment")
    else:
        raise ValueError(f"Unsupported analytical card: {card}")

    for include_examples in (True, False):
        task = compose_card_prompt(
            instruction, examples, include_examples=include_examples
        )
        task_separator = "\n\n" + task
        if remainder.endswith(task_separator):
            state = remainder[: -len(task_separator)]
            if card == "exact":
                if not (
                    state.startswith("Reference time: ")
                    and "\nRole: " in state
                    and "\nUser message: " in state
                    and "\nRecent messages:\n" in state
                    and "\nNeed type meanings:\n" in state
                ):
                    raise ValueError("Exact-recall state does not match the production builder")
            elif not (
                state.startswith('<source_message role="')
                and state.endswith("</assistant_history>")
            ):
                raise ValueError("Sentiment state does not match the production builder")
            return metadata, state, task
    raise ValueError("Card task does not match either production example setting")


def orient_request(request: LLMCompletionRequest) -> LLMCompletionRequest:
    """Move one captured production card's rules to its native choice field."""
    questions = request.choice_questions
    if not questions or len(questions) != 1 or len(request.messages) != 2:
        raise ValueError("GLiNER native input requires one captured choice card")
    system, user = request.messages
    if system.role != "system" or user.role != "user":
        raise ValueError("Captured card messages do not match production roles")
    question_id, question = next(iter(questions.items()))
    if question_id == "context_reuse":
        intro, state = _split_context_reuse(user.content)
        instructions = "\n\n".join((system.content, intro, question.instructions))
    elif question_id in {"exact", "outcome_sentiment"}:
        card = "exact" if question_id == "exact" else "sentiment"
        expected_card = card
        if (
            request.metadata.get("need_detection_card")
            if card == "exact"
            else request.metadata.get("consequence_detection_card")
        ) != expected_card:
            raise ValueError("Captured card metadata does not match its question")
        metadata, state, task = _split_analytical_card(user.content, card=card)
        instructions = "\n\n".join(
            (system.content, metadata, task, question.instructions)
        )
    else:
        raise ValueError(f"Unsupported GLiNER native input question: {question_id}")

    return request.model_copy(
        update={
            "messages": [LLMMessage(role="user", content=state)],
            "choice_questions": {
                question_id: question.model_copy(update={"instructions": instructions})
            },
        }
    )
