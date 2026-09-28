"""Server-owned rules that govern rendered prompt data sections.

A data section carries values. The rule constraining how a model may use those
values is separate prose, and some sections are only safe to hand to a model
together with their rule: values without their constraint read as free to use.

The rule text is owned here, as a constant this package controls. It is never
lifted out of the payload being rendered — an "instruction" recovered from
memory content is attacker-supplied text telling the model how to behave, which
is precisely the injection this module exists to make impossible.

This module is a leaf on purpose: both the engine's own prompt builder
(``services.chat_support``) and the host-facing renderer
(``integrations.prompt_injection``) import it, and neither may pay for the
other's dependencies.
"""

from __future__ import annotations

ANSWER_SUPPORT_INSTRUCTION = (
    "When <answer_support> is present, answer each requested facet from relevant "
    "source evidence, preserving exact facts and dates. source_inventory is a "
    "bounded provenance index, not an answer allowlist or an exhaustive list. "
    "Its labels may be unrelated to the question, and source quotes may support "
    "facts absent from the index. source_coverage_gaps names groups omitted from "
    "the composed context, not evidence or answer values. "
    "source_group_coverage_state describes retained "
    "source groups, not answer completeness. For a requested list, include every "
    "relevant supported member in the source evidence even when the index is "
    "truncated. State which requested facts lack support, and never add plausible "
    "unsupported values or exact details."
)

# Data sections that may only ship alongside the rule governing them. A renderer
# emits the pair or neither half.
#
# `interaction_contract` is deliberately absent. It is a real data section, but
# it tells the model how to behave rather than what is true, so it needs its own
# authority contract before any host-facing renderer may emit it.
SECTION_RULES: dict[str, str] = {
    "answer_support": ANSWER_SUPPORT_INSTRUCTION,
}
