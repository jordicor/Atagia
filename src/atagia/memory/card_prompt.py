"""Shared helper for assembling card prompts with an optional examples block.

A "card" prompt is split into two parts: the instruction (what to read and
decide, plus inline term definitions and the output format) and the few-shot
demonstration block (concrete input -> output pairs). The demonstrations are
gated by ``include_examples`` so a deployment can drop them per component.

Few-shot demonstrations reliably help small/local models (they pin the output
format and answer space) but can hurt larger or reasoning models, so the
examples block is toggleable without maintaining a second prompt set. The
small/local target class is the default, so examples are included by default.
"""

from __future__ import annotations

EXAMPLES_HEADER = "Examples:"


def compose_card_prompt(
    instruction: str,
    examples: str | None,
    *,
    include_examples: bool,
) -> str:
    """Join a card instruction with its demonstration block when enabled.

    ``examples`` holds only the demonstration pairs; the ``Examples:`` header is
    added here so every card renders the block identically. When examples are
    disabled or absent, only the instruction is returned.
    """
    instruction = instruction.rstrip()
    if include_examples and examples and examples.strip():
        return f"{instruction}\n\n{EXAMPLES_HEADER}\n{examples.strip()}"
    return instruction
