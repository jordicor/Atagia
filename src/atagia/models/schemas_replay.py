"""Schemas for replay, comparison, grounding, and dataset export."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from atagia.models.schemas_memory import (
    ComposedContext,
    DetectedNeed,
    RetrievalPlan,
    RetrievalSufficiencyDiagnostic,
    RetrievalTrace,
    ScoredCandidate,
)


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

    model_config = ConfigDict(extra="forbid")

    skip_need_detection: bool = False
    skip_applicability_scoring: bool = False
    skip_contract_memory: bool = False
    skip_workspace_rollup: bool = False
    force_all_scopes: bool = False
    skip_belief_revision: bool = False
    skip_compaction: bool = False
    disable_context_cache: bool = False
    composer_strategy: Literal["score_first", "budgeted_marginal"] | None = None
    override_retrieval_params: dict[str, Any] | None = None


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
    anonymization_mode: ExportAnonymizationMode = ExportAnonymizationMode.RAW
