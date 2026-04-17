"""Final context composition within token budgets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from typing import Any

from atagia.core.clock import Clock
from atagia.memory.policy_manifest import ResolvedPolicy
from atagia.models.schemas_memory import ComposedContext, QueryType, ScoredCandidate



@dataclass(frozen=True, slots=True)
class _SelectionProfile:
    diversity_weight: float
    richness_weight: float
    pool_extra: int


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
        query_text: str | None = None,
        query_type: QueryType = "default",
        exact_recall_mode: bool = False,
    ) -> ComposedContext:
        coerced_candidates = [
            candidate
            for candidate in (
                self._coerce_scored_candidate(candidate)
                for candidate in scored_candidates
            )
            if str(candidate.memory_object.get("canonical_text", "")).strip()
        ]
        # Wave 1 batch 2 (1-D): when exact recall is flagged, push L0
        # evidence ahead of L1/L2 summaries before any diversity pass.
        candidates = sorted(
            coerced_candidates,
            key=lambda candidate: (
                self._exact_recall_level_priority(candidate, exact_recall_mode),
                -candidate.final_score,
                candidate.memory_id,
            ),
        )
        candidates = self._selection_order(
            candidates,
            max_items=resolved_policy.retrieval_params.final_context_items,
            query_text=query_text,
            query_type=query_type,
        )
        candidate_by_id = {
            candidate.memory_id: candidate
            for candidate in candidates
        }
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
        selected_ids: set[str] = set()
        memory_lines: list[str] = []
        max_items = resolved_policy.retrieval_params.final_context_items
        for candidate in candidates:
            if len(selected) >= max_items:
                break
            if remaining_budget <= 0:
                break
            if candidate.memory_id in selected_ids:
                continue

            if self._is_hierarchical_summary_candidate(candidate):
                conflicting_l0 = self._find_conflicting_fresher_l0(
                    candidate,
                    candidates,
                    selected_ids,
                )
                if conflicting_l0 is not None:
                    remaining_budget = self._append_candidate_if_possible(
                        conflicting_l0,
                        selected=selected,
                        selected_ids=selected_ids,
                        memory_lines=memory_lines,
                        remaining_budget=remaining_budget,
                        max_items=max_items,
                    )
                    continue

                supporting_l0 = self._supporting_l0_candidate(
                    candidate,
                    candidate_by_id,
                    selected_ids,
                )
                if supporting_l0 is None:
                    continue
                if supporting_l0.memory_id not in selected_ids:
                    required_items = 2
                    required_tokens = (
                        self.estimate_tokens(self._format_memory_entry(len(selected) + 1, supporting_l0))
                        + self.estimate_tokens(self._format_memory_entry(len(selected) + 2, candidate))
                    )
                    if len(selected) + required_items > max_items or required_tokens > remaining_budget:
                        continue
                    remaining_budget = self._append_candidate_if_possible(
                        supporting_l0,
                        selected=selected,
                        selected_ids=selected_ids,
                        memory_lines=memory_lines,
                        remaining_budget=remaining_budget,
                        max_items=max_items,
                    )
                remaining_budget = self._append_candidate_if_possible(
                    candidate,
                    selected=selected,
                    selected_ids=selected_ids,
                    memory_lines=memory_lines,
                    remaining_budget=remaining_budget,
                    max_items=max_items,
                )
                continue

            remaining_budget = self._append_candidate_if_possible(
                candidate,
                selected=selected,
                selected_ids=selected_ids,
                memory_lines=memory_lines,
                remaining_budget=remaining_budget,
                max_items=max_items,
            )

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

    @classmethod
    def _selection_order(
        cls,
        candidates: list[ScoredCandidate],
        *,
        max_items: int,
        query_text: str | None,
        query_type: QueryType,
    ) -> list[ScoredCandidate]:
        if len(candidates) <= 1 or not query_text or max_items <= 0:
            return candidates

        profile = cls._selection_profile(query_type)
        if (
            profile.diversity_weight <= 0.0
            and profile.richness_weight <= 0.0
        ):
            return candidates

        pool_size = min(
            len(candidates),
            max(max_items * 3, max_items + profile.pool_extra),
        )
        pool = list(candidates[:pool_size])
        remainder = candidates[pool_size:]
        query_tokens = cls._content_tokens(query_text)
        repeated_pool_tokens = cls._repeated_pool_tokens(pool)
        selected: list[ScoredCandidate] = []

        while pool:
            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(pool):
                redundancy = max(
                    cls._candidate_similarity(
                        candidate,
                        chosen,
                        ignored_tokens=repeated_pool_tokens,
                    )
                    for chosen in selected
                ) if selected else 0.0
                richness = cls._candidate_richness(
                    candidate,
                    query_tokens=query_tokens,
                    repeated_pool_tokens=repeated_pool_tokens,
                )
                selection_score = (
                    float(candidate.final_score)
                    + (profile.richness_weight * richness)
                    - (profile.diversity_weight * redundancy)
                )
                if selection_score > best_score:
                    best_index = index
                    best_score = selection_score
                    continue
                if math.isclose(selection_score, best_score):
                    incumbent = pool[best_index]
                    if (
                        float(candidate.final_score),
                        candidate.memory_id,
                    ) > (
                        float(incumbent.final_score),
                        incumbent.memory_id,
                    ):
                        best_index = index
                        best_score = selection_score
            selected.append(pool.pop(best_index))

        return [*selected, *remainder]

    @staticmethod
    def _selection_profile(query_type: QueryType) -> _SelectionProfile:
        if query_type == "broad_list":
            return _SelectionProfile(
                diversity_weight=0.32,
                richness_weight=0.12,
                pool_extra=24,
            )
        if query_type == "temporal":
            return _SelectionProfile(
                diversity_weight=0.03,
                richness_weight=0.01,
                pool_extra=10,
            )
        if query_type == "slot_fill":
            return _SelectionProfile(
                diversity_weight=0.18,
                richness_weight=0.07,
                pool_extra=16,
            )
        return _SelectionProfile(
            diversity_weight=0.08,
            richness_weight=0.03,
            pool_extra=12,
        )

    @classmethod
    def _candidate_similarity(
        cls,
        candidate: ScoredCandidate,
        other: ScoredCandidate,
        *,
        ignored_tokens: set[str],
    ) -> float:
        candidate_tokens = cls._content_tokens(
            str(candidate.memory_object.get("canonical_text", "")),
            ignored_tokens=ignored_tokens,
        )
        other_tokens = cls._content_tokens(
            str(other.memory_object.get("canonical_text", "")),
            ignored_tokens=ignored_tokens,
        )
        if not candidate_tokens or not other_tokens:
            return 0.0
        overlap = candidate_tokens & other_tokens
        union = candidate_tokens | other_tokens
        similarity = len(overlap) / len(union)
        if cls._candidate_source_window(candidate) == cls._candidate_source_window(other):
            similarity += 0.15
        return min(1.0, similarity)

    @classmethod
    def _candidate_richness(
        cls,
        candidate: ScoredCandidate,
        *,
        query_tokens: set[str],
        repeated_pool_tokens: set[str],
    ) -> float:
        ignored_tokens = {*query_tokens, *repeated_pool_tokens}
        candidate_tokens = cls._content_tokens(
            str(candidate.memory_object.get("canonical_text", "")),
            ignored_tokens=ignored_tokens,
        )
        if not candidate_tokens:
            return 0.0
        richness = min(1.0, sum(len(token) for token in candidate_tokens) / 40.0)
        if str(candidate.memory_object.get("object_type")) in {"belief", "interaction_contract", "state_snapshot"}:
            return richness * 0.7
        return richness

    @staticmethod
    def _candidate_source_window(candidate: ScoredCandidate) -> tuple[str, str] | None:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return None
        start = str(payload_json.get("source_message_window_start_occurred_at") or "").strip()
        end = str(payload_json.get("source_message_window_end_occurred_at") or "").strip()
        if not start and not end:
            return None
        return (start, end)

    @classmethod
    def _content_tokens(
        cls,
        text: str,
        *,
        ignored_tokens: set[str] | None = None,
    ) -> set[str]:
        ignored = ignored_tokens or set()
        return {
            normalized
            for token in cls._tokenize(text)
            if (normalized := cls._normalize_token(token))
            and normalized not in ignored
            and len(normalized) > 1
        }

    @staticmethod
    def _normalize_token(token: str) -> str:
        normalized = token.strip("_")
        if not normalized:
            return ""
        if not any(character.isalnum() for character in normalized):
            return ""
        return normalized

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens: list[str] = []
        current: list[str] = []
        for character in text.lower():
            if character.isalnum() or character == "_":
                current.append(character)
                continue
            if current:
                tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))
        return tokens

    @classmethod
    def _repeated_pool_tokens(cls, candidates: list[ScoredCandidate]) -> set[str]:
        token_counts: dict[str, int] = {}
        for candidate in candidates:
            for token in cls._content_tokens(str(candidate.memory_object.get("canonical_text", ""))):
                token_counts[token] = token_counts.get(token, 0) + 1
        return {
            token
            for token, count in token_counts.items()
            if count >= 2
        }

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

    @staticmethod
    def _exact_recall_level_priority(
        candidate: ScoredCandidate,
        exact_recall_mode: bool,
    ) -> int:
        """Return the exact-recall hierarchy bucket for a candidate.

        Lower is better. Concrete L0 evidence (including raw message
        windows) sits in bucket 0, while episode/thematic summaries
        drop to bucket 1. Outside exact-recall mode all candidates are
        treated equally so ordering stays score-driven.
        """
        if not exact_recall_mode:
            return 0
        memory_object = candidate.memory_object
        object_type = str(memory_object.get("object_type"))
        if object_type != "summary_view":
            return 0
        payload_json = memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return 1
        try:
            hierarchy_level = int(payload_json.get("hierarchy_level", 0))
        except (TypeError, ValueError):
            return 1
        return 0 if hierarchy_level == 0 else 1

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
        payload_json = memory_object.get("payload_json") or {}
        is_conversation_chunk = (
            memory_object.get("object_type") == "summary_view"
            and isinstance(payload_json, dict)
            and payload_json.get("summary_kind") == "conversation_chunk"
        )
        metadata_parts = [
            f"{memory_object.get('object_type')}",
            f"confidence: {confidence:.2f}",
            f"scope: {memory_object.get('scope')}",
        ]
        if is_conversation_chunk:
            source_window_start = payload_json.get("source_message_window_start_occurred_at")
            source_window_end = payload_json.get("source_message_window_end_occurred_at")
            if source_window_start or source_window_end:
                metadata_parts.append(
                    f"source_window: {source_window_start or '?'} to {source_window_end or '?'}"
                )
        valid_from = memory_object.get("valid_from")
        valid_to = memory_object.get("valid_to")
        if valid_from or valid_to:
            metadata_parts.append(f"valid: {valid_from or '?'} to {valid_to or '?'}")
        if candidate.resolved_date:
            metadata_parts.append(f"resolved_date: {candidate.resolved_date}")
        lines = [
            f"{index}. ({', '.join(metadata_parts)})\n"
            f"   {memory_object.get('canonical_text', '')}"
        ]
        if is_conversation_chunk:
            excerpt = ContextComposer._conversation_chunk_excerpt(payload_json)
            if excerpt:
                lines.append(f"   source_excerpt: {excerpt}")
        return "\n".join(lines)

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

    @classmethod
    def _append_candidate_if_possible(
        cls,
        candidate: ScoredCandidate,
        *,
        selected: list[ScoredCandidate],
        selected_ids: set[str],
        memory_lines: list[str],
        remaining_budget: int,
        max_items: int,
    ) -> int:
        if candidate.memory_id in selected_ids or remaining_budget <= 0 or len(selected) >= max_items:
            return remaining_budget
        candidate_block = cls._format_memory_entry(len(selected) + 1, candidate)
        candidate_tokens = cls.estimate_tokens(candidate_block)
        if candidate_tokens > remaining_budget:
            return remaining_budget
        memory_lines.append(candidate_block)
        selected.append(candidate)
        selected_ids.add(candidate.memory_id)
        return remaining_budget - candidate_tokens

    @staticmethod
    def _is_hierarchical_summary_candidate(candidate: ScoredCandidate) -> bool:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if candidate.memory_object.get("object_type") != "summary_view" or not isinstance(payload_json, dict):
            return False
        return int(payload_json.get("hierarchy_level", -1)) in {1, 2}

    @classmethod
    def _supporting_l0_candidate(
        cls,
        candidate: ScoredCandidate,
        candidate_by_id: dict[str, ScoredCandidate],
        selected_ids: set[str],
    ) -> ScoredCandidate | None:
        source_ids = cls._candidate_source_ids(candidate)
        if not source_ids:
            return None
        supporting_candidate = cls._best_supporting_l0_candidate(
            source_ids=source_ids,
            candidate_by_id=candidate_by_id,
            selected_ids=selected_ids,
            visited={candidate.memory_id},
            allow_selected=False,
        )
        if supporting_candidate is not None:
            return supporting_candidate
        return cls._best_supporting_l0_candidate(
            source_ids=source_ids,
            candidate_by_id=candidate_by_id,
            selected_ids=selected_ids,
            visited={candidate.memory_id},
            allow_selected=True,
        )

    @staticmethod
    def _candidate_source_ids(candidate: ScoredCandidate) -> list[str]:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return []
        return [
            str(item).strip()
            for item in payload_json.get("source_object_ids", [])
            if str(item).strip()
        ]

    @classmethod
    def _best_supporting_l0_candidate(
        cls,
        *,
        source_ids: list[str],
        candidate_by_id: dict[str, ScoredCandidate],
        selected_ids: set[str],
        visited: set[str],
        allow_selected: bool,
    ) -> ScoredCandidate | None:
        direct_support: list[ScoredCandidate] = []
        recursive_support: list[ScoredCandidate] = []
        for source_id in source_ids:
            supporting_candidate = candidate_by_id.get(source_id)
            if supporting_candidate is None:
                continue
            if supporting_candidate.memory_object.get("object_type") != "summary_view":
                if allow_selected or supporting_candidate.memory_id not in selected_ids:
                    direct_support.append(supporting_candidate)
                continue
            if supporting_candidate.memory_id in visited:
                continue
            nested_source_ids = cls._candidate_source_ids(supporting_candidate)
            if not nested_source_ids:
                continue
            recursive_support_candidate = cls._best_supporting_l0_candidate(
                source_ids=nested_source_ids,
                candidate_by_id=candidate_by_id,
                selected_ids=selected_ids,
                visited={*visited, supporting_candidate.memory_id},
                allow_selected=allow_selected,
            )
            if recursive_support_candidate is not None:
                recursive_support.append(recursive_support_candidate)
        all_support = [*direct_support, *recursive_support]
        if not all_support:
            return None
        return sorted(
            all_support,
            key=lambda item: (-item.final_score, item.memory_id),
        )[0]

    @classmethod
    def _find_conflicting_fresher_l0(
        cls,
        candidate: ScoredCandidate,
        all_candidates: list[ScoredCandidate],
        selected_ids: set[str],
    ) -> ScoredCandidate | None:
        payload_json = candidate.memory_object.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return None
        claim_signatures = payload_json.get("source_claim_signatures", [])
        if not isinstance(claim_signatures, list) or not claim_signatures:
            return None
        summary_updated_at = cls._parse_candidate_datetime(candidate.memory_object.get("updated_at"))
        source_object_ids = {
            str(item).strip()
            for item in payload_json.get("source_object_ids", [])
            if str(item).strip()
        }
        expected_values_by_key: dict[str, set[str]] = {}
        for signature in claim_signatures:
            if not isinstance(signature, dict):
                continue
            claim_key = str(signature.get("claim_key") or "").strip()
            if not claim_key:
                continue
            expected_values_by_key.setdefault(claim_key, set()).add(
                json.dumps(signature.get("claim_value"), ensure_ascii=False, sort_keys=True)
            )
        if not expected_values_by_key:
            return None

        conflicting: list[ScoredCandidate] = []
        for other in all_candidates:
            if other.memory_id in selected_ids or other.memory_id in source_object_ids:
                continue
            if other.memory_object.get("object_type") != "belief":
                continue
            other_payload = other.memory_object.get("payload_json") or {}
            if not isinstance(other_payload, dict):
                continue
            claim_key = str(other_payload.get("claim_key") or "").strip()
            if claim_key not in expected_values_by_key:
                continue
            claim_value = json.dumps(other_payload.get("claim_value"), ensure_ascii=False, sort_keys=True)
            if claim_value in expected_values_by_key[claim_key]:
                continue
            other_updated_at = cls._parse_candidate_datetime(other.memory_object.get("updated_at"))
            if summary_updated_at is not None and other_updated_at is not None and other_updated_at <= summary_updated_at:
                continue
            conflicting.append(other)
        if not conflicting:
            return None
        return sorted(conflicting, key=lambda item: (-item.final_score, item.memory_id))[0]

    @staticmethod
    def _parse_candidate_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    @staticmethod
    def _conversation_chunk_excerpt(payload_json: dict[str, Any]) -> str:
        excerpt_messages = payload_json.get("source_excerpt_messages")
        if not isinstance(excerpt_messages, list):
            return ""
        rendered: list[str] = []
        for item in excerpt_messages:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            role = str(item.get("role", "")).strip() or "unknown"
            occurred_at = str(item.get("occurred_at", "")).strip()
            prefix = role
            if occurred_at:
                prefix += f" @ {occurred_at}"
            rendered.append(f"{prefix}: {text}")
        return " | ".join(rendered)
