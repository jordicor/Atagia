"""Schemas for replay, comparison, grounding, and dataset export."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from atagia.models.schemas_memory import (
    AdaptiveGateStatus,
    ComposedContext,
    DetectedNeed,
    IntimacyBoundary,
    MemoryDependence,
    RetrievalPlan,
    RetrievalSufficiencyDiagnostic,
    RetrievalTrace,
    ScoredCandidate,
)

# Upper bound on the LLM coverage-expansion sub-query knob, and the pipeline's
# default when the knob is not overridden. It lives beside the knob's validated
# range rather than at the use site so the bound that rejects a requested value
# and the bound the engine enforces are the same object.
LLM_COVERAGE_MAX_SUBQUERIES: Final[int] = 3

# Inclusive integer bounds for every override_retrieval_params key whose value
# is a bounded number; ``None`` as the upper bound means unbounded above. Four
# of them (fts_limit, vector_limit, rerank_top_k, final_context_items) mirror
# RetrievalParams fields and repeat that model's own Field bounds, and
# privacy_ceiling repeats RetrievalPlan's -- rejecting the value here is what
# stops an override from building a model that violates its own constraints.
# tests/models/test_override_retrieval_params_validation.py re-derives those
# bounds from the models and fails on drift.
_OVERRIDE_RETRIEVAL_PARAM_BOUNDS: dict[str, tuple[int, int | None]] = {
    "fts_limit": (0, None),
    "vector_limit": (0, None),
    "rerank_top_k": (1, None),
    "final_context_items": (1, None),
    "max_candidates": (0, None),
    "max_context_items": (1, None),
    "privacy_ceiling": (0, 3),
    "context_budget_tokens": (1, None),
    "transcript_budget_tokens": (1, None),
    "llm_coverage_candidate_limit": (1, None),
    "llm_coverage_max_subqueries": (1, LLM_COVERAGE_MAX_SUBQUERIES),
}

# The one recognized override that is a flag rather than a bounded integer.
_OVERRIDE_RETRIEVAL_PARAM_BOOL_KEYS: frozenset[str] = frozenset(
    {"allow_private_sensitivity"}
)

# Single source of truth for the override_retrieval_params keys the retrieval
# pipeline actually reads. Four of them mirror RetrievalParams fields (applied
# in RetrievalPipeline._override_policy, which loops over
# RetrievalParams.model_fields); the rest are consumed explicitly by the
# pipeline (_build_plan, _override_policy, _cap_explicit_final_context_items,
# _llm_coverage_candidate_limit, _llm_coverage_max_subqueries). Deriving the set
# from the two domain tables above makes "key the engine reads" and "values the
# engine can honor" the same declaration, so a key can never be accepted without
# a validated domain. The boundary validator on AblationConfig, and the no-drift
# test that re-derives the set from the pipeline consumption sites, both use it.
RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS: frozenset[str] = (
    frozenset(_OVERRIDE_RETRIEVAL_PARAM_BOUNDS) | _OVERRIDE_RETRIEVAL_PARAM_BOOL_KEYS
)


def _bounds_text(bounds: tuple[int, int | None]) -> str:
    minimum, maximum = bounds
    return f">= {minimum}" if maximum is None else f"{minimum}..{maximum}"


def _override_value_rejection(key: str, value: Any) -> str | None:
    """Describe why the engine cannot honor ``value`` for ``key``, else None."""
    if key in _OVERRIDE_RETRIEVAL_PARAM_BOOL_KEYS:
        if isinstance(value, bool):
            return None
        return f"{key}={value!r} (valid values: true, false)"
    bounds = _OVERRIDE_RETRIEVAL_PARAM_BOUNDS[key]
    # bool is an int subclass, so a flag would otherwise pass as 0 or 1.
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{key}={value!r} (valid values: integers {_bounds_text(bounds)})"
    minimum, maximum = bounds
    if value < minimum or (maximum is not None and value > maximum):
        return f"{key}={value} (valid range: {_bounds_text(bounds)})"
    return None


class GroundingLevel(str, Enum):
    """Grounding classification for a selected memory."""

    GROUNDED = "grounded"
    DERIVED = "derived"
    INFERRED = "inferred"
    SUMMARY = "summary"


class ExportAnonymizationMode(str, Enum):
    """Available admin export anonymization modes."""

    RAW = "raw"
    STRICT = "strict"
    READABLE = "readable"


class ConversationExportKind(str, Enum):
    """Top-level artifact type for conversation export."""

    RAW_REPLAY = "raw_replay"
    ANONYMIZED_PROJECTION = "anonymized_projection"


class AblationConfig(BaseModel):
    """Optional switches that modify replay-time retrieval behavior."""

    # Frozen because this is THE validated boundary for retrieval overrides: a
    # config whose fields could be reassigned after construction would let a
    # caller install a value the validator below never saw, which is the defect
    # this class exists to prevent. It also matches every neighboring schema
    # (RetrievalParams, RetrievalProfileManifest, ResolvedRetrievalPolicy).
    # model_copy(update=...) still works, which is how the trusted-evaluation
    # merge and the benchmark presets derive variants.
    model_config = ConfigDict(extra="forbid", frozen=True)

    privacy_enforcement: Literal["enforce", "audit_only", "off"] = "enforce"
    skip_need_detection: bool = False
    skip_applicability_scoring: bool = False
    # Fusion dedupe is ON by default; this switch exists for A/B
    # probes and ablation studies only.
    skip_fusion_dedupe: bool = False
    applicability_gate_mode: Literal["off", "shadow", "enforced"] | None = None
    skip_contract_memory: bool = False
    skip_workspace_rollup: bool = False
    force_all_scopes: bool = False
    skip_belief_revision: bool = False
    skip_compaction: bool = False
    disable_context_cache: bool = False
    enable_llm_coverage_expansion: bool = False
    enable_evidence_obligation_coverage: bool = True
    enable_evidence_packets: bool = True
    enable_final_answer_evidence_pack: bool = False
    composer_strategy: Literal["score_first", "budgeted_marginal"] | None = None
    # Read-only by construction. ``frozen=True`` stops the FIELD from being
    # reassigned, but a plain dict behind it could still be mutated in place
    # after validation -- and the retrieval trace copies this exact object into
    # ``applied_override_retrieval_params``, so an injected key would have been
    # recorded as applied while no consumer ever read it. Storing an immutable
    # view of a private copy is what makes "applied == requested" structural
    # instead of a comment asserting it. The field serializer below dumps it
    # back as a plain dict, so every model_dump/model_dump_json round trip is
    # unchanged.
    override_retrieval_params: Mapping[str, Any] | None = None
    context_envelope_budget_tokens: int | None = Field(default=None, gt=0)
    context_envelope_ratios: dict[str, float] | None = None

    @field_serializer("override_retrieval_params")
    def _serialize_override_retrieval_params(
        self, value: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        return None if value is None else dict(value)

    @field_validator("override_retrieval_params")
    @classmethod
    def _validate_override_retrieval_params(
        cls, value: Mapping[str, Any] | None
    ) -> Mapping[str, Any] | None:
        """Reject keys the pipeline never reads and values it cannot honor.

        Unknown or typo'd keys were silently ignored, so a run could vary a knob
        the engine did not consume. Reject them at the earliest boundary and name
        the offenders alongside the recognized set.

        Values get the same treatment, and for the same reason: an unusable
        value used to be silently clamped (a requested privacy_ceiling of 99 ran
        as 3), which is the value-level version of the defect the key check
        exists to stop. Rejecting instead of clamping makes the recorded
        override structurally equal to the requested one -- there is no longer a
        normalization step that could diverge from what the engine applies -- and
        it matches ContextBudgetAboveEnvelopeError, which already raises on an
        above-envelope context_budget_tokens rather than capping it.

        The validated mapping is returned as an immutable view of a private
        copy: the caller's dict cannot leak later writes into it, and nothing
        downstream can add a key this validator never saw. An empty mapping is
        wrapped too -- ``{}`` is exactly the case where a post-construction
        insert would have been invisible.
        """
        if value is None:
            return None
        unknown = sorted(set(value) - RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS)
        if unknown:
            recognized = ", ".join(sorted(RECOGNIZED_OVERRIDE_RETRIEVAL_PARAM_KEYS))
            raise ValueError(
                "Unknown override_retrieval_params key(s): "
                f"{', '.join(unknown)}. Recognized keys: {recognized}"
            )
        rejected = [
            rejection
            for key in sorted(value)
            if (rejection := _override_value_rejection(key, value[key])) is not None
        ]
        if rejected:
            raise ValueError(
                "Unusable override_retrieval_params value(s): " + "; ".join(rejected)
            )
        return MappingProxyType(dict(value))


class PipelineResult(BaseModel):
    """Reusable retrieval pipeline output."""

    model_config = ConfigDict(extra="forbid")

    detected_needs: list[DetectedNeed] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan
    raw_candidates: list[dict[str, Any]] = Field(default_factory=list)
    scored_candidates: list[ScoredCandidate] = Field(default_factory=list)
    candidate_custody: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_sufficiency: RetrievalSufficiencyDiagnostic | None = None
    composed_context: ComposedContext
    current_contract: dict[str, dict[str, Any]] = Field(default_factory=dict)
    user_state: dict[str, Any] = Field(default_factory=dict)
    stage_timings: dict[str, float] = Field(default_factory=dict)
    trace: RetrievalTrace | None = None
    small_corpus_mode: bool = False
    degraded_mode: bool = False
    # Adaptive retrieval gate observability. The default is shadow (gate
    # computed a classification but took no action), which keeps every existing
    # construction valid; the classification is optional because degraded or
    # gate-free paths may not produce one.
    adaptive_gate_status: AdaptiveGateStatus = AdaptiveGateStatus.OFF_SHADOW
    adaptive_gate_classification: MemoryDependence | None = None


class ScoreDelta(BaseModel):
    """Per-memory score change between original and replay."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    original_score: float
    replay_score: float
    delta: float


class RetrievalComparison(BaseModel):
    """Comparison between an original retrieval event and a replay."""

    model_config = ConfigDict(extra="forbid")

    memories_in_both: list[str] = Field(default_factory=list)
    memories_only_original: list[str] = Field(default_factory=list)
    memories_only_replay: list[str] = Field(default_factory=list)
    score_deltas: list[ScoreDelta] = Field(default_factory=list)
    contract_block_changed: bool = False
    workspace_block_changed: bool = False
    memory_block_changed: bool = False
    state_block_changed: bool = False
    original_items_count: int = Field(ge=0)
    replay_items_count: int = Field(ge=0)
    overlap_ratio: float = Field(ge=0.0, le=1.0)
    original_total_tokens: int = Field(ge=0)
    replay_total_tokens: int = Field(ge=0)


class ReplayResult(BaseModel):
    """Replay output for a single retrieval event."""

    model_config = ConfigDict(extra="forbid")

    original_event_id: str
    replay_pipeline_result: PipelineResult
    comparison: RetrievalComparison
    ablation_config: dict[str, Any] | None = None


class GroundingItem(BaseModel):
    """Grounding analysis for one selected memory."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    canonical_text: str
    object_type: str
    source_kind: str
    maya_score: float
    grounding_level: GroundingLevel
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GroundingReport(BaseModel):
    """Grounding analysis over a composed context."""

    model_config = ConfigDict(extra="forbid")

    items: list[GroundingItem] = Field(default_factory=list)
    grounded_ratio: float = Field(ge=0.0, le=1.0)
    avg_maya_score: float = Field(ge=0.0)
    high_maya_items: list[str] = Field(default_factory=list)


class ExportedMessage(BaseModel):
    """Serializable message export row."""

    model_config = ConfigDict(extra="forbid")

    message_id: str
    seq: int
    role: str
    content: str
    occurred_at: str | None = None
    created_at: str | None = None


class ExportAnonymizedEntity(BaseModel):
    """Safe placeholder metadata for an anonymized export."""

    model_config = ConfigDict(extra="forbid")

    placeholder: str
    readable_label: str


class ExportAnonymizationSummary(BaseModel):
    """Safe summary of export anonymization behavior."""

    model_config = ConfigDict(extra="forbid")

    mode: ExportAnonymizationMode
    applied: bool = True
    entity_count: int = Field(ge=0)
    entities: list[ExportAnonymizedEntity] = Field(default_factory=list)


class ExportedRetrievalTrace(BaseModel):
    """Serializable retrieval trace export row."""

    model_config = ConfigDict(extra="forbid")

    retrieval_event_id: str
    request_message_seq: int
    detected_needs: list[str] = Field(default_factory=list)
    retrieval_plan: dict[str, Any] = Field(default_factory=dict)
    selected_memory_ids: list[str] = Field(default_factory=list)
    scored_candidates: list[dict[str, Any]] = Field(default_factory=list)
    context_view: dict[str, Any] = Field(default_factory=dict)
    outcome: dict[str, Any] = Field(default_factory=dict)


class ConversationExport(BaseModel):
    """Conversation export payload for replay or anonymized projection use."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    user_id: str
    assistant_mode_id: str
    export_kind: ConversationExportKind = ConversationExportKind.RAW_REPLAY
    replay_compatible: bool = True
    workspace_id: str | None = None
    messages: list[ExportedMessage] = Field(default_factory=list)
    retrieval_traces: list[ExportedRetrievalTrace] | None = None
    intimacy_boundary_counts: dict[str, int] = Field(default_factory=dict)
    exported_at: str | None = None
    anonymization: ExportAnonymizationSummary | None = None


class ReplayEventRequest(BaseModel):
    """Admin replay request for a single retrieval event."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    ablation: AblationConfig | None = None


class ReplayConversationRequest(BaseModel):
    """Admin replay request for a whole conversation."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    ablation: AblationConfig | None = None
    message_limit: int | None = Field(default=None, ge=1)


class GroundingRequest(BaseModel):
    """Admin grounding analysis request."""

    model_config = ConfigDict(extra="forbid")

    user_id: str


class ConversationExportRequest(BaseModel):
    """Admin conversation export request."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    include_retrieval_traces: bool = True
    include_intimacy_context: bool = False
    anonymization_mode: ExportAnonymizationMode = ExportAnonymizationMode.RAW
