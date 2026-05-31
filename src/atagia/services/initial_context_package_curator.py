"""Background-only LLM curation for prepared initial-context packages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atagia.core import json_utils
from atagia.core.config import Settings
from atagia.models.schemas_initial_context_package import InitialContextPackageProfileItem
from atagia.services.chat_support import estimate_tokens
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage
from atagia.services.model_resolution import resolve_component_model

logger = logging.getLogger(__name__)

_PROFILE_STATUS_VALUES = {"current", "historical", "superseded", "ambiguous"}
_MAX_CANDIDATE_TEXT_CHARS = 900


@dataclass(frozen=True, slots=True)
class InitialContextPackageCurationResult:
    """Validated curated orientation items plus non-fatal build warnings."""

    items: list[InitialContextPackageProfileItem]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class _CurationCandidate:
    candidate_id: str
    source_kind: str
    text: str
    status: str
    source_refs: list[dict[str, Any]]
    scope_json: dict[str, Any]
    coordinate_visibility_json: dict[str, Any]
    freshness_json: dict[str, Any]
    salience: float | None


class _CuratedItem(BaseModel):
    """One source-grounded orientation item proposed by the model."""

    model_config = ConfigDict(extra="ignore")

    candidate_ids: list[str] = Field(default_factory=list, min_length=1)
    text: str = Field(min_length=1)
    status: Literal["current", "historical", "superseded", "ambiguous"] = "current"
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    reason_category: str = "curated_orientation"

    @field_validator("candidate_ids")
    @classmethod
    def normalize_candidate_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in values:
            value = str(raw).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        if not normalized:
            raise ValueError("candidate_ids must include at least one source")
        return normalized

    @field_validator("text", "reason_category")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("curated text cannot be blank")
        return normalized


class _CurationPlan(BaseModel):
    """Structured response for package curation."""

    model_config = ConfigDict(extra="ignore")

    items: list[_CuratedItem] = Field(default_factory=list)
    nothing_to_add: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_nothing_to_add(self) -> "_CurationPlan":
        if self.nothing_to_add and self.items:
            raise ValueError("nothing_to_add cannot be true when items are present")
        return self


class InitialContextPackageCurator:
    """Curate a compact, grounded orientation block during package refresh."""

    def __init__(
        self,
        *,
        llm_client: LLMClient[Any],
        settings: Settings | None = None,
    ) -> None:
        self._llm_client = llm_client
        resolved_settings = settings or Settings.from_env()
        self._model = resolve_component_model(
            resolved_settings,
            "initial_context_package_curation",
        )
        self._max_items = resolved_settings.initial_context_package_curated_max_items
        self._max_output_tokens = (
            resolved_settings.initial_context_package_curation_max_output_tokens
        )
        self._block_budget_tokens = (
            resolved_settings.initial_context_package_curated_block_max_tokens
        )

    async def curate(
        self,
        *,
        user_id: str,
        package_kind: str,
        retrieval_profile_id: str,
        coordinate_complete: bool,
        profile_items: list[InitialContextPackageProfileItem],
        current_state_block: str,
        current_state_refs: list[dict[str, Any]],
        conversation_summary_block: str,
        summary_refs: list[dict[str, Any]],
        working_topic_block: str,
        topic_refs: list[dict[str, Any]],
        recent_verbatim_seed: list[dict[str, Any]],
    ) -> InitialContextPackageCurationResult:
        if not coordinate_complete:
            return InitialContextPackageCurationResult(
                [],
                ["curation_skipped_coordinate_incomplete"],
            )
        candidates = self._candidates(
            profile_items=profile_items,
            current_state_block=current_state_block,
            current_state_refs=current_state_refs,
            conversation_summary_block=conversation_summary_block,
            summary_refs=summary_refs,
            working_topic_block=working_topic_block,
            topic_refs=topic_refs,
            recent_verbatim_seed=recent_verbatim_seed,
        )
        if not candidates:
            return InitialContextPackageCurationResult([], ["curation_skipped_no_candidates"])
        try:
            plan = await self._request_plan(
                user_id=user_id,
                package_kind=package_kind,
                retrieval_profile_id=retrieval_profile_id,
                candidates=candidates,
            )
        except Exception as exc:
            logger.warning(
                "initial_context_package_curation_failed",
                extra={
                    "user_id": user_id,
                    "package_kind": package_kind,
                    "error": str(exc),
                },
            )
            return InitialContextPackageCurationResult(
                [],
                [f"curation_failed:{exc.__class__.__name__}"],
            )
        items, warnings = self._validated_items(plan, candidates)
        if not items:
            warnings.append("curation_produced_no_valid_items")
        return InitialContextPackageCurationResult(items, warnings)

    async def _request_plan(
        self,
        *,
        user_id: str,
        package_kind: str,
        retrieval_profile_id: str,
        candidates: list[_CurationCandidate],
    ) -> _CurationPlan:
        request = LLMCompletionRequest(
            model=self._model,
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "Curate source-grounded initial context for an assistant. "
                        "Return JSON only. Do not invent facts, do not answer the user, "
                        "and do not use any information outside the supplied candidates."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=self._prompt(
                        package_kind=package_kind,
                        retrieval_profile_id=retrieval_profile_id,
                        candidates=candidates,
                    ),
                ),
            ],
            max_output_tokens=self._max_output_tokens,
            response_schema=_CurationPlan.model_json_schema(),
            metadata={
                "user_id": user_id,
                "purpose": "initial_context_package_curation",
                "package_kind": package_kind,
                "retrieval_profile_id": retrieval_profile_id,
            },
        )
        return await self._llm_client.complete_structured(request, _CurationPlan)

    def _prompt(
        self,
        *,
        package_kind: str,
        retrieval_profile_id: str,
        candidates: list[_CurationCandidate],
    ) -> str:
        candidate_payload = [
            {
                "candidate_id": candidate.candidate_id,
                "source_kind": candidate.source_kind,
                "status": candidate.status,
                "salience": candidate.salience,
                "text": _compact(candidate.text, _MAX_CANDIDATE_TEXT_CHARS),
            }
            for candidate in candidates
        ]
        return "\n".join(
            [
                f"Package kind: {package_kind}",
                f"Retrieval profile: {retrieval_profile_id}",
                f"Select up to {self._max_items} orientation items.",
                (
                    "Prefer facts that would help the model quickly recognize who/what "
                    "this context is about, including pivotal changes, recent state, "
                    "and stable preferences. Preserve temporal status: current, "
                    "historical, superseded, or ambiguous."
                ),
                (
                    "If cited candidates conflict, preserve the ambiguity instead of "
                    "flattening the contradiction into one asserted truth; use status "
                    "ambiguous when that is the most faithful orientation."
                ),
                (
                    "Do not turn old emotional states into present orientation unless "
                    "the cited candidates explicitly show they explain the present; "
                    "otherwise keep them historical/superseded or skip them."
                ),
                (
                    "Each returned item must cite candidate_ids from the list. If the "
                    "best result would merely repeat low-value facts, set nothing_to_add=true."
                ),
                (
                    "Keep all curated text concise and directly supported by cited candidates."
                ),
                "<candidates_json>",
                json_utils.dumps(candidate_payload, sort_keys=True),
                "</candidates_json>",
            ]
        )

    def _validated_items(
        self,
        plan: _CurationPlan,
        candidates: list[_CurationCandidate],
    ) -> tuple[list[InitialContextPackageProfileItem], list[str]]:
        if plan.nothing_to_add:
            return [], ["curation_nothing_to_add"]
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        curated: list[InitialContextPackageProfileItem] = []
        warnings: list[str] = []
        seen_texts: set[str] = set()
        for index, item in enumerate(plan.items[: self._max_items]):
            source_candidates = [
                by_id[candidate_id]
                for candidate_id in item.candidate_ids
                if candidate_id in by_id
            ]
            if not source_candidates:
                warnings.append("curation_dropped_ungrounded_item")
                continue
            text = _compact(item.text, 360)
            text_key = text.casefold()
            if text_key in seen_texts:
                warnings.append("curation_dropped_duplicate_item")
                continue
            source_refs = _merge_source_refs(source_candidates)
            if not source_refs:
                warnings.append("curation_dropped_item_without_refs")
                continue
            status = _validated_status(item.status, source_candidates, warnings)
            curated.append(
                InitialContextPackageProfileItem(
                    item_id=f"curated:{index}:{_stable_item_hash(text, item.candidate_ids)}",
                    text=text,
                    reason_category=_compact(item.reason_category, 64),
                    source_refs=source_refs,
                    scope_json=_merge_mapping(source_candidates, "scope_json"),
                    coordinate_visibility_json=_merge_mapping(
                        source_candidates,
                        "coordinate_visibility_json",
                    ),
                    freshness_json={
                        "curated": True,
                        "candidate_ids": item.candidate_ids,
                        "source_freshness": [
                            candidate.freshness_json for candidate in source_candidates
                        ],
                    },
                    status=status,
                    salience=float(item.salience),
                )
            )
            seen_texts.add(text_key)
            if estimate_tokens(self.render_block(curated)) > self._block_budget_tokens:
                curated.pop()
                warnings.append("curation_trimmed_block_budget")
                break
        return curated, warnings

    @staticmethod
    def render_block(items: list[InitialContextPackageProfileItem]) -> str:
        if not items:
            return ""
        lines = [
            "[Curated Initial Orientation]",
            (
                "Background orientation curated during refresh; live evidence "
                "and recent transcript outrank this block."
            ),
        ]
        for item in items:
            refs = ", ".join(
                str(ref.get("memory_id") or ref.get("message_id") or ref.get("source_kind"))
                for ref in item.source_refs[:3]
            )
            prefix = f"[{item.status}] " if item.status != "current" else ""
            suffix = f" (source: {refs})" if refs else ""
            lines.append(f"- {prefix}{item.text}{suffix}")
        return "\n".join(lines)

    def _candidates(
        self,
        *,
        profile_items: list[InitialContextPackageProfileItem],
        current_state_block: str,
        current_state_refs: list[dict[str, Any]],
        conversation_summary_block: str,
        summary_refs: list[dict[str, Any]],
        working_topic_block: str,
        topic_refs: list[dict[str, Any]],
        recent_verbatim_seed: list[dict[str, Any]],
    ) -> list[_CurationCandidate]:
        candidates = [
            _CurationCandidate(
                candidate_id=item.item_id,
                source_kind="profile_item",
                text=item.text,
                status=item.status,
                source_refs=item.source_refs,
                scope_json=item.scope_json,
                coordinate_visibility_json=item.coordinate_visibility_json,
                freshness_json=item.freshness_json,
                salience=item.salience,
            )
            for item in profile_items
        ]
        candidates.extend(
            self._block_candidates(
                "current_state",
                current_state_block,
                current_state_refs,
                status="current",
            )
        )
        candidates.extend(
            self._block_candidates(
                "conversation_summary",
                conversation_summary_block,
                summary_refs,
                status="historical",
            )
        )
        candidates.extend(
            self._block_candidates(
                "working_topic",
                working_topic_block,
                topic_refs,
                status="current",
            )
        )
        for seed in recent_verbatim_seed:
            if bool(seed.get("skip_by_default")):
                continue
            message_id = str(seed.get("message_id") or "").strip()
            text = str(seed.get("text") or "").strip()
            if not message_id or not text:
                continue
            candidates.append(
                _CurationCandidate(
                    candidate_id=f"recent_seed:{message_id}",
                    source_kind="recent_message",
                    text=text,
                    status="current",
                    source_refs=[
                        {
                            "source_kind": "message",
                            "message_id": message_id,
                            "conversation_id": seed.get("conversation_id"),
                            "seq": seed.get("seq"),
                        }
                    ],
                    scope_json={},
                    coordinate_visibility_json={},
                    freshness_json={"occurred_at": seed.get("occurred_at")},
                    salience=None,
                )
            )
        return candidates

    @staticmethod
    def _block_candidates(
        source_kind: str,
        block: str,
        refs: list[dict[str, Any]],
        *,
        status: str,
    ) -> list[_CurationCandidate]:
        text = block.strip()
        if not text or not refs:
            return []
        return [
            _CurationCandidate(
                candidate_id=f"block:{source_kind}",
                source_kind=source_kind,
                text=text,
                status=status,
                source_refs=refs,
                scope_json={},
                coordinate_visibility_json={},
                freshness_json={},
                salience=None,
            )
        ]


def _merge_source_refs(candidates: list[_CurationCandidate]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        for ref in candidate.source_refs:
            key = json_utils.dumps(ref, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(ref))
    return merged[:8]


def _merge_mapping(candidates: list[_CurationCandidate], attr: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for candidate in candidates:
        value = getattr(candidate, attr)
        if isinstance(value, Mapping):
            for key, raw in value.items():
                if raw is not None and key not in merged:
                    merged[str(key)] = raw
    return merged


def _strongest_status(candidates: list[_CurationCandidate]) -> str:
    for status in ("ambiguous", "superseded", "historical"):
        if any(candidate.status == status for candidate in candidates):
            return status
    return "current"


def _validated_status(
    proposed_status: str,
    candidates: list[_CurationCandidate],
    warnings: list[str],
) -> str:
    if proposed_status not in _PROFILE_STATUS_VALUES:
        warnings.append("curation_corrected_invalid_status")
        return _strongest_status(candidates)
    if proposed_status == "current" and all(
        candidate.status != "current" for candidate in candidates
    ):
        warnings.append("curation_corrected_noncurrent_status")
        return _strongest_status(candidates)
    return proposed_status


def _stable_item_hash(text: str, candidate_ids: list[str]) -> str:
    raw = "\x1f".join([text, *candidate_ids])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _compact(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 1)].rstrip() + "..."
