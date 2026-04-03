"""Memory domain schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryObjectType(str, Enum):
    EVIDENCE = "evidence"
    BELIEF = "belief"
    INTERACTION_CONTRACT = "interaction_contract"
    STATE_SNAPSHOT = "state_snapshot"
    CONSEQUENCE_CHAIN = "consequence_chain"
    SUMMARY_VIEW = "summary_view"


class MemoryScope(str, Enum):
    GLOBAL_USER = "global_user"
    ASSISTANT_MODE = "assistant_mode"
    WORKSPACE = "workspace"
    CONVERSATION = "conversation"
    EPHEMERAL_SESSION = "ephemeral_session"


class MemorySourceKind(str, Enum):
    VERBATIM = "verbatim"
    EXTRACTED = "extracted"
    INFERRED = "inferred"
    SUMMARIZED = "summarized"
    COMPOSED = "composed"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"
    REVIEW_REQUIRED = "review_required"


class SummaryViewKind(str, Enum):
    CONVERSATION_CHUNK = "conversation_chunk"
    WORKSPACE_ROLLUP = "workspace_rollup"
    CONTEXT_VIEW = "context_view"


class NeedTrigger(str, Enum):
    AMBIGUITY = "ambiguity"
    CONTRADICTION = "contradiction"
    FOLLOW_UP_FAILURE = "follow_up_failure"
    LOOP = "loop"
    HIGH_STAKES = "high_stakes"
    MODE_SHIFT = "mode_shift"
    FRUSTRATION = "frustration"
    SENSITIVE_CONTEXT = "sensitive_context"
    UNDER_SPECIFIED_REQUEST = "under_specified_request"


class ConsequenceSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AssistantModeId(str, Enum):
    CODING_DEBUG = "coding_debug"
    RESEARCH_DEEP_DIVE = "research_deep_dive"
    COMPANION = "companion"
    BRAINSTORM = "brainstorm"
    BIOGRAPHICAL_INTERVIEW = "biographical_interview"
    GENERAL_QA = "general_qa"


def _ensure_unique_values(values: list[Any]) -> list[Any]:
    """Reject duplicate values while preserving order semantics."""
    if len(values) != len(set(values)):
        raise ValueError("List values must be unique")
    return values


class RetrievalParams(BaseModel):
    """Retrieval tuning knobs carried by an assistant mode policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fts_limit: int = Field(ge=0)
    vector_limit: int = Field(ge=0)
    graph_hops: int = Field(ge=0)
    rerank_top_k: int = Field(gt=0)
    final_context_items: int = Field(gt=0)


class ContextCachePolicy(BaseModel):
    """Adaptive context-cache settings attached to an assistant mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_ttl_seconds: int = Field(gt=0)
    sync_threshold: float = Field(gt=0.0, le=1.0)
    max_messages_without_refresh: int = Field(ge=1)
    max_minutes_without_refresh: int = Field(ge=1)


class AssistantModeManifest(BaseModel):
    """Validated assistant mode manifest loaded from JSON files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assistant_mode_id: AssistantModeId
    display_name: str = Field(min_length=1)
    cross_chat_allowed: bool
    allowed_scopes: list[MemoryScope] = Field(min_length=1)
    preferred_memory_types: list[MemoryObjectType] = Field(min_length=1)
    need_triggers: list[NeedTrigger] = Field(default_factory=list)
    contract_dimensions_priority: list[str] = Field(min_length=1)
    privacy_ceiling: int = Field(ge=0, le=3)
    context_budget_tokens: int = Field(gt=0)
    transcript_budget_tokens: int = Field(gt=0)
    retrieval_params: RetrievalParams
    context_cache_policy: ContextCachePolicy
    prompt_hash: str | None = Field(default=None, exclude=True)

    @field_validator("allowed_scopes")
    @classmethod
    def validate_allowed_scopes(cls, values: list[MemoryScope]) -> list[MemoryScope]:
        return _ensure_unique_values(values)

    @field_validator("preferred_memory_types")
    @classmethod
    def validate_preferred_memory_types(
        cls,
        values: list[MemoryObjectType],
    ) -> list[MemoryObjectType]:
        return _ensure_unique_values(values)

    @field_validator("need_triggers")
    @classmethod
    def validate_need_triggers(cls, values: list[NeedTrigger]) -> list[NeedTrigger]:
        return _ensure_unique_values(values)

    @field_validator("contract_dimensions_priority")
    @classmethod
    def validate_contract_dimensions_priority(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Contract dimensions must be non-empty strings")
        return _ensure_unique_values(values)


class ExtractionContextMessage(BaseModel):
    """Recent conversation message passed into the extractor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: str


class ExtractionConversationContext(BaseModel):
    """Execution context for a single extraction run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    conversation_id: str
    source_message_id: str
    workspace_id: str | None = None
    assistant_mode_id: str
    recent_messages: list[ExtractionContextMessage] = Field(default_factory=list)


class ExtractedMemoryBase(BaseModel):
    """Shared fields returned by the extraction model."""

    model_config = ConfigDict(extra="forbid")

    canonical_text: str = Field(min_length=1)
    scope: MemoryScope
    confidence: float = Field(ge=0.0, le=1.0)
    source_kind: MemorySourceKind
    privacy_level: int = Field(ge=0, le=3)
    payload: dict[str, Any] = Field(default_factory=dict)


class ExtractedEvidence(ExtractedMemoryBase):
    """Grounded observation extracted from a message."""


class ExtractedBelief(ExtractedMemoryBase):
    """Belief candidate inferred from the message."""

    claim_key: str = Field(min_length=1)
    claim_value: str = Field(min_length=1)


class ExtractedContractSignal(ExtractedMemoryBase):
    """Collaboration preference signal extracted from the message."""


class ExtractedStateUpdate(ExtractedMemoryBase):
    """State snapshot update extracted from the message."""


class ExtractionResult(BaseModel):
    """Structured output returned by the memory extractor."""

    model_config = ConfigDict(extra="forbid")

    evidences: list[ExtractedEvidence] = Field(default_factory=list)
    beliefs: list[ExtractedBelief] = Field(default_factory=list)
    contract_signals: list[ExtractedContractSignal] = Field(default_factory=list)
    state_updates: list[ExtractedStateUpdate] = Field(default_factory=list)
    # Hook for mode-shift detection in Step 7.
    mode_guess: str | None = None
    nothing_durable: bool = False

    @model_validator(mode="after")
    def validate_nothing_durable_consistency(self) -> "ExtractionResult":
        if self.nothing_durable and (
            self.evidences or self.beliefs or self.contract_signals or self.state_updates
        ):
            raise ValueError("nothing_durable=true but extraction lists are non-empty")
        return self


class ContractSignal(BaseModel):
    """Contract preference signal extracted from a message."""

    model_config = ConfigDict(extra="forbid")

    canonical_text: str = Field(min_length=1)
    dimension_name: str = Field(min_length=1)
    value_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    scope: MemoryScope
    source_kind: MemorySourceKind = MemorySourceKind.INFERRED
    privacy_level: int = Field(ge=0, le=3, default=1)

    @field_validator("dimension_name")
    @classmethod
    def validate_dimension_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("dimension_name must be non-empty")
        return normalized


class ContractProjectionResult(BaseModel):
    """Structured output returned by the contract projector."""

    model_config = ConfigDict(extra="forbid")

    signals: list[ContractSignal] = Field(default_factory=list)
    nothing_durable: bool = False

    @model_validator(mode="after")
    def validate_nothing_durable_consistency(self) -> "ContractProjectionResult":
        if self.nothing_durable and self.signals:
            raise ValueError("nothing_durable=true but contract signals are non-empty")
        return self


class DetectedNeed(BaseModel):
    """Need signal detected from the active message and recent context."""

    model_config = ConfigDict(extra="forbid")

    need_type: NeedTrigger
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reasoning must be non-empty")
        return normalized


class ConsequenceSignal(BaseModel):
    """Detected report of an outcome caused by prior assistant action."""

    model_config = ConfigDict(extra="forbid")

    is_consequence: bool
    action_description: str = ""
    outcome_description: str = ""
    outcome_sentiment: ConsequenceSentiment = ConsequenceSentiment.NEUTRAL
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    likely_action_message_id: str | None = None

    @field_validator("action_description", "outcome_description")
    @classmethod
    def validate_descriptions(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_consequence_confidence(self) -> "ConsequenceSignal":
        if self.is_consequence and self.confidence < 0.1:
            raise ValueError("is_consequence=true requires confidence >= 0.1")
        return self


class ConsequenceChainResult(BaseModel):
    """Identifiers returned after building a consequence chain."""

    model_config = ConfigDict(extra="forbid")

    chain_id: str
    action_memory_id: str
    outcome_memory_id: str
    tendency_belief_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievalPlan(BaseModel):
    """Deterministic retrieval plan built before candidate search."""

    model_config = ConfigDict(extra="forbid")

    assistant_mode_id: str
    workspace_id: str | None = None
    conversation_id: str
    fts_queries: list[str] = Field(default_factory=list)
    scope_filter: list[MemoryScope] = Field(default_factory=list)
    status_filter: list[MemoryStatus] = Field(default_factory=list)
    vector_limit: int = Field(ge=0, default=0)
    max_candidates: int = Field(ge=0)
    max_context_items: int = Field(gt=0)
    privacy_ceiling: int = Field(ge=0, le=3)
    require_evidence_regrounding: bool = False
    need_driven_boosts: dict[NeedTrigger, float] = Field(default_factory=dict)
    skip_retrieval: bool = False

    @field_validator("fts_queries")
    @classmethod
    def validate_fts_queries(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return _ensure_unique_values(normalized)

    @field_validator("scope_filter")
    @classmethod
    def validate_scope_filter(cls, values: list[MemoryScope]) -> list[MemoryScope]:
        return _ensure_unique_values(values)

    @field_validator("status_filter")
    @classmethod
    def validate_status_filter(cls, values: list[MemoryStatus]) -> list[MemoryStatus]:
        return _ensure_unique_values(values)


class ScoredCandidate(BaseModel):
    """Scored retrieval candidate ready for context composition."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    memory_object: dict[str, Any]
    llm_applicability: float = Field(ge=0.0, le=1.0)
    retrieval_score: float = Field(ge=0.0, le=1.0)
    vitality_boost: float = Field(ge=0.0, le=1.0)
    confirmation_boost: float = Field(ge=0.0, le=1.0)
    need_boost: float = Field(ge=0.0)
    penalty: float = Field(ge=0.0)
    final_score: float


class ComposedContext(BaseModel):
    """Ephemeral prompt-ready context assembled for a single response."""

    model_config = ConfigDict(extra="forbid")

    contract_block: str = ""
    workspace_block: str = ""
    memory_block: str = ""
    state_block: str = ""
    selected_memory_ids: list[str] = Field(default_factory=list)
    total_tokens_estimate: int = Field(ge=0)
    budget_tokens: int = Field(ge=0)
    items_included: int = Field(ge=0)
    items_dropped: int = Field(ge=0)


class MemoryObject(BaseModel):
    """Canonical memory object."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    assistant_mode_id: str | None = None
    object_type: MemoryObjectType
    scope: MemoryScope
    canonical_text: str
    payload_json: dict[str, Any] = Field(default_factory=dict)
    source_kind: MemorySourceKind
    confidence: float = 0.5
    stability: float = 0.5
    vitality: float = 0.0
    maya_score: float = 0.0
    privacy_level: int = 0
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime
    updated_at: datetime


class ContractDimensionCurrent(BaseModel):
    """Projection row for an interaction contract dimension."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    assistant_mode_id: str | None = None
    scope: MemoryScope
    dimension_name: str
    value_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    source_memory_id: str
    updated_at: datetime


class RetrievalEvent(BaseModel):
    """Trace of a retrieval operation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    conversation_id: str
    request_message_id: str
    response_message_id: str | None = None
    assistant_mode_id: str
    retrieval_plan_json: dict[str, Any]
    selected_memory_ids_json: list[str] = Field(default_factory=list)
    context_view_json: dict[str, Any] = Field(default_factory=dict)
    outcome_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
