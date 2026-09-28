"""Benchmark-only compact and focused source-reference challengers."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from atagia.core.source_references import SourceReference, SourceReferenceCatalog
from atagia.memory.extraction_cards import CandidateDraft
from atagia.memory.source_quote_selector import (
    _anchor_options,
    _messages,
    _pair,
    _questions,
    _validate_candidates,
)
from atagia.models.schemas_decisions import ChoiceQuestion
from atagia.models.schemas_memory import MemoryEvidenceSupportKind
from atagia.services.llm_client import LLMClient
from atagia.services.model_resolution import parse_model_spec


Mode = Literal["compact", "focused"]
_COMPACT_BLOCK_SIZE = 254
_FOCUSED_BLOCK_SIZE = 64
_MAX_BLOCKS = 254


def _compact_options(refs: tuple[str, ...]) -> dict[str, str | None]:
    return {"none": _anchor_options(())["none"], **dict.fromkeys(refs)}


def _block_context(
    catalog: SourceReferenceCatalog,
    blocks: tuple[tuple[str, ...], ...],
    source_context: str,
    source_block: str,
) -> tuple[str, dict[str, str]]:
    rendered = catalog.render()
    options = {"none": _anchor_options(())["none"]}
    for index, block in enumerate(blocks, start=1):
        options[f"b{index}"] = f"Block [{block[0]}] through [{block[-1]}]"
        rendered = rendered.replace(
            f"[{block[0]}]",
            f'<block id="b{index}" first="{block[0]}" last="{block[-1]}"/>[{block[0]}]',
            1,
        )
    context = source_context.replace(
        source_block, f"<message_text>\n{rendered}\n</message_text>", 1
    )
    return context, options


def _focused_context(
    catalog: SourceReferenceCatalog,
    blocks: tuple[tuple[str, ...], ...],
    selected: dict[str, tuple[int, int]],
    source_context: str,
    source_block: str,
) -> str:
    """Keep every chosen block and explicitly identify each omitted gap."""
    included = {
        index
        for first, last in selected.values()
        for index in range(first, last + 1)
    }
    if not included:
        raise ValueError("Focused refinement needs at least one selected block")
    ranges: list[tuple[int, int]] = []
    for index in sorted(included):
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], index)
        else:
            ranges.append((index, index))

    rendered = catalog.render()
    starts = [rendered.index(f"[{block[0]}]") for block in blocks]
    pieces: list[str] = []
    previous_last = -1
    for first, last in ranges:
        if first > previous_last + 1:
            pieces.append(
                f'<omitted_source_blocks first="b{previous_last + 2}" last="b{first}"/>'
            )
        begin = 0 if first == 0 else starts[first]
        end = len(rendered) if last + 1 == len(blocks) else starts[last + 1]
        pieces.append(
            f'<included_source_region first="b{first + 1}" last="b{last + 1}">'
            f"{rendered[begin:end]}</included_source_region>"
        )
        previous_last = last
    if previous_last < len(blocks) - 1:
        pieces.append(
            f'<omitted_source_blocks first="b{previous_last + 2}" last="b{len(blocks)}"/>'
        )
    focused_source = "\n".join(pieces)
    return source_context.replace(
        source_block, f"<message_text>\n{focused_source}\n</message_text>", 1
    )


async def select_source_references(
    llm_client: LLMClient[Any],
    *,
    model: str,
    source_text: str,
    candidates: tuple[CandidateDraft, ...],
    support_kinds: dict[str, str],
    source_context: str,
    source_catalog: SourceReferenceCatalog | None = None,
    metadata: dict[str, Any] | None = None,
    dispatch_semaphore: asyncio.Semaphore | None = None,
    mode: Mode,
) -> dict[str, SourceReference | None]:
    """Run a frozen reference-only challenger with native finite choices."""
    if mode not in ("compact", "focused"):
        raise ValueError("Unknown source selector challenger mode")
    if parse_model_spec(model).provider_name != "typesafe":
        raise ValueError("Native source selection requires a finite-choice model")
    _validate_candidates(candidates)
    if set(support_kinds) != {item.candidate_id for item in candidates} or any(
        value not in {kind.value for kind in MemoryEvidenceSupportKind}
        for value in support_kinds.values()
    ):
        raise ValueError("Source selection requires each candidate's support assessment")
    catalog = source_catalog or SourceReferenceCatalog(source_text)
    if catalog.source_text != source_text:
        raise ValueError("Source catalog must match the extraction message")
    source_block = f"<message_text>\n{catalog.render()}\n</message_text>"
    if source_context.count(source_block) != 1:
        raise ValueError("Source context must contain exactly the referenced source")
    refs = tuple(anchor.reference_id for anchor in catalog.anchors)
    if not refs:
        return {candidate.candidate_id: None for candidate in candidates}
    request_metadata = {
        **(metadata or {}),
        "purpose": "memory_extraction_source_reference_selector",
        "challenger_mode": mode,
    }

    async def complete(
        questions: dict[str, ChoiceQuestion],
        context: str,
        stage: str,
        instruction: str = "",
    ) -> dict[str, str]:
        answers = await llm_client.complete_choice_questions(
            model=model,
            messages=_messages(context, instruction),
            questions=questions,
            metadata={**request_metadata, "stage": stage},
            dispatch_semaphore=dispatch_semaphore,
        )
        if set(answers) != set(questions) or any(
            value not in questions[key].criteria for key, value in answers.items()
        ):
            raise ValueError("Source selector returned missing or unknown answers")
        return answers

    if len(refs) <= _COMPACT_BLOCK_SIZE:
        answers = await complete(
            _questions(candidates, support_kinds, _compact_options(refs), "reference"),
            source_context,
            "anchors",
        )
        return {
            candidate.candidate_id: catalog.resolve(*pair)
            if (pair := _pair(answers, candidate.candidate_id))
            else None
            for candidate in candidates
        }

    block_size = _COMPACT_BLOCK_SIZE if mode == "compact" else _FOCUSED_BLOCK_SIZE
    blocks = tuple(
        refs[start : start + block_size] for start in range(0, len(refs), block_size)
    )
    if len(blocks) > _MAX_BLOCKS:
        raise ValueError("Source exceeds the native selector's block capacity")
    block_context, block_options = _block_context(catalog, blocks, source_context, source_block)
    answers = await complete(
        _questions(candidates, support_kinds, block_options, "block"),
        block_context,
        "blocks",
        " Blocks are consecutive source ranges. First locate each boundary's block.",
    )
    selected: dict[str, tuple[int, int]] = {}
    result: dict[str, SourceReference | None] = {}
    for candidate in candidates:
        pair = _pair(answers, candidate.candidate_id)
        if pair is None:
            result[candidate.candidate_id] = None
            continue
        first, last = (int(value[1:]) - 1 for value in pair)
        if first > last:
            raise ValueError("Source selector returned reversed blocks")
        selected[candidate.candidate_id] = (first, last)
    if not selected:
        return result

    active = tuple(item for item in candidates if item.candidate_id in selected)
    questions = {
        f"{candidate.candidate_id}.{side}": ChoiceQuestion(
            instructions=(
                f"Choose the exact {side} reference for {candidate.candidate_id} "
                f"inside the selected block b{selected[candidate.candidate_id][index] + 1}. "
                "Retain every premise and qualification needed to support it."
            ),
            criteria=_compact_options(blocks[selected[candidate.candidate_id][index]]),
        )
        for candidate in active
        for index, side in enumerate(("start", "end"))
    }
    refinement_context = (
        block_context
        if mode == "compact"
        else _focused_context(catalog, blocks, selected, source_context, source_block)
    )
    instruction = (
        " Refine the fixed block selections to exact boundaries. The passage may "
        "cross blocks. Each question restricts choices to its selected block."
    )
    if mode == "focused":
        instruction += (
            " Included regions retain their original reference IDs and source order. "
            "Omitted-source markers are gaps, not adjacent source text."
        )
    answers = await complete(questions, refinement_context, "anchors", instruction)
    for candidate in active:
        pair = _pair(answers, candidate.candidate_id)
        if pair is None:
            raise ValueError("Source boundaries contradict the selected support blocks")
        result[candidate.candidate_id] = catalog.resolve(*pair)
    return result
