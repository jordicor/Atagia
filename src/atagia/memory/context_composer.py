"""Final context composition within token budgets."""

from __future__ import annotations

import json
import math
from typing import Any

from atagia.core.clock import Clock
from atagia.memory.policy_manifest import ResolvedPolicy
from atagia.models.schemas_memory import ComposedContext, ScoredCandidate


class ContextComposer:
    """Formats scored memories into a prompt-ready context view."""

    CONTRACT_BLOCK_RATIO = 0.35
    WORKSPACE_BLOCK_RATIO = 0.08

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def compose(
        self,
        scored_candidates: list[ScoredCandidate | dict[str, Any]],
        current_contract: dict[str, dict[str, Any]],
        user_state: dict[str, Any] | None,
        resolved_policy: ResolvedPolicy,
        conversation_messages: list[dict[str, Any]],
        workspace_rollup: dict[str, Any] | None = None,
    ) -> ComposedContext:
        del conversation_messages  # Phase 1 keeps the raw transcript outside these composed blocks.

        candidates = sorted(
            (
                candidate
                for candidate in (
                    self._coerce_scored_candidate(candidate)
                    for candidate in scored_candidates
                )
                if str(candidate.memory_object.get("canonical_text", "")).strip()
            ),
            key=lambda candidate: (-candidate.final_score, candidate.memory_id),
        )
        budget_tokens = resolved_policy.context_budget_tokens
        remaining_budget = budget_tokens
        contract_block, remaining_budget = self._budget_block(
            self._format_contract_block(current_contract, resolved_policy),
            total_budget=budget_tokens,
            remaining_budget=remaining_budget,
            ratio=self.CONTRACT_BLOCK_RATIO,
            min_tokens=1,
            header="[Interaction Contract]",
        )
        workspace_block, remaining_budget = self._budget_block(
            self._format_workspace_block(workspace_rollup),
            total_budget=budget_tokens,
            remaining_budget=remaining_budget,
            ratio=self.WORKSPACE_BLOCK_RATIO,
            min_tokens=0,
            header="[Workspace Context]",
        )
        state_block, remaining_budget = self._budget_block(
            self._format_state_block(user_state),
            total_budget=remaining_budget,
            remaining_budget=remaining_budget,
            ratio=1.0,
            min_tokens=0,
            header="[Current User State]",
        )
        memory_header = "[Retrieved Memories]\n"
        if candidates:
            header_tokens = self.estimate_tokens(memory_header)
            if remaining_budget >= header_tokens:
                remaining_budget -= header_tokens
            else:
                remaining_budget = 0

        selected: list[ScoredCandidate] = []
        memory_lines: list[str] = []
        max_items = resolved_policy.retrieval_params.final_context_items
        for candidate in candidates:
            if len(selected) >= max_items:
                break
            if remaining_budget <= 0:
                break
            candidate_block = self._format_memory_entry(len(selected) + 1, candidate)
            candidate_tokens = self.estimate_tokens(candidate_block)
            if candidate_tokens > remaining_budget:
                continue
            memory_lines.append(candidate_block)
            selected.append(candidate)
            remaining_budget -= candidate_tokens

        memory_block = (
            memory_header + "\n".join(memory_lines)
            if memory_lines
            else ""
        )
        total_tokens_estimate = (
            self.estimate_tokens(contract_block)
            + self.estimate_tokens(workspace_block)
            + self.estimate_tokens(memory_block)
            + self.estimate_tokens(state_block)
        )
        if total_tokens_estimate > budget_tokens:
            raise RuntimeError("Context composition exceeded the resolved token budget")
        items_included = len(selected)
        items_dropped = len(candidates) - items_included
        return ComposedContext(
            contract_block=contract_block,
            workspace_block=workspace_block,
            memory_block=memory_block,
            state_block=state_block,
            selected_memory_ids=[candidate.memory_id for candidate in selected],
            total_tokens_estimate=total_tokens_estimate,
            budget_tokens=budget_tokens,
            items_included=items_included,
            items_dropped=items_dropped,
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    @staticmethod
    def _coerce_scored_candidate(candidate: ScoredCandidate | dict[str, Any]) -> ScoredCandidate:
        if isinstance(candidate, ScoredCandidate):
            return candidate
        return ScoredCandidate.model_validate(candidate)

    def _format_contract_block(
        self,
        current_contract: dict[str, dict[str, Any]],
        resolved_policy: ResolvedPolicy,
    ) -> str:
        ordered_dimensions = list(resolved_policy.contract_dimensions_priority)
        ordered_dimensions.extend(
            sorted(
                dimension
                for dimension in current_contract
                if dimension not in resolved_policy.contract_dimensions_priority
            )
        )
        lines = ["[Interaction Contract]"]
        for dimension in ordered_dimensions:
            value = current_contract.get(dimension)
            if value is None:
                continue
            lines.append(f"- {dimension}: {self._format_contract_value(value)}")
        return "\n".join(lines)

    @staticmethod
    def _format_contract_value(value: dict[str, Any]) -> str:
        if not value:
            return "unspecified"
        label = value.get("label")
        if label is None:
            extras = {key: item for key, item in value.items() if key not in {"score", "confidence"}}
            label = json.dumps(extras or value, ensure_ascii=False, sort_keys=True)
        rendered = str(label)
        confidence = value.get("confidence", value.get("score"))
        if isinstance(confidence, (int, float)):
            rendered += f" (confidence: {float(confidence):.2f})"
        return rendered

    @classmethod
    def _format_workspace_block(
        cls,
        workspace_rollup: dict[str, Any] | None,
        *,
        budget_tokens: int | None = None,
    ) -> str:
        if not workspace_rollup:
            return ""
        summary_text = str(workspace_rollup.get("summary_text", "")).strip()
        if not summary_text:
            return ""
        block = f"[Workspace Context]\n{summary_text}"
        if budget_tokens is None:
            return block
        max_tokens = max(1, math.ceil(budget_tokens * cls.WORKSPACE_BLOCK_RATIO))
        return cls._truncate_block(
            block,
            max_tokens=max_tokens,
            header="[Workspace Context]",
        )

    @staticmethod
    def _format_memory_entry(index: int, candidate: ScoredCandidate) -> str:
        memory_object = candidate.memory_object
        confidence = float(memory_object.get("confidence", 0.0))
        return (
            f"{index}. ({memory_object.get('object_type')}, confidence: {confidence:.2f}, "
            f"scope: {memory_object.get('scope')})\n"
            f"   {memory_object.get('canonical_text', '')}"
        )

    @staticmethod
    def _format_state_block(user_state: dict[str, Any] | None) -> str:
        if not user_state:
            return ""
        lines = ["[Current User State]"]
        for key, value in sorted(user_state.items()):
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                rendered = str(value)
            lines.append(f"- {key}: {rendered}")
        return "\n".join(lines)

    @classmethod
    def _budget_block(
        cls,
        text: str,
        *,
        total_budget: int,
        remaining_budget: int,
        ratio: float,
        min_tokens: int,
        header: str,
    ) -> tuple[str, int]:
        if not text or remaining_budget <= 0:
            return "", remaining_budget
        allocated_tokens = min(
            remaining_budget,
            max(min_tokens, math.ceil(total_budget * ratio)),
        )
        block = cls._truncate_block(
            text,
            max_tokens=allocated_tokens,
            header=header,
        )
        return block, max(0, remaining_budget - cls.estimate_tokens(block))

    @classmethod
    def _truncate_block(cls, text: str, *, max_tokens: int, header: str = "") -> str:
        if max_tokens <= 0 or not text:
            return ""
        if cls.estimate_tokens(text) <= max_tokens:
            return text

        max_chars = max_tokens * 4
        if not header:
            if max_chars <= 3:
                return text[:max_chars]
            return f"{text[: max_chars - 3].rstrip()}..."

        if max_chars <= len(header):
            return header[:max_chars].rstrip()

        suffix = "..."
        content = text[len(header):].strip()
        available_chars = max(0, max_chars - len(header) - len(suffix))
        if available_chars == 0:
            return header.rstrip()
        truncated = content[:available_chars].rstrip()
        return f"{header}{truncated}{suffix}"
