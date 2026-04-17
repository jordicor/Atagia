"""API request and response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atagia.core.timestamps import normalize_optional_timestamp
from atagia.models.schemas_evaluation import MetricName
from atagia.models.schemas_memory import (
    ComposedContext,
    MemoryScope,
    VerbatimPinStatus,
    VerbatimPinTargetKind,
)


class ChatMessageInput(BaseModel):
    """Minimal input payload for a chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateUserRequest(BaseModel):
    """Create a user in Atagia."""

    model_config = ConfigDict(extra="forbid")

    user_id: str


class CreateConversationRequest(BaseModel):
    """Create a conversation in Atagia."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    conversation_id: str | None = None
    assistant_mode_id: str | None = None
    workspace_id: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateWorkspaceRequest(BaseModel):
    """Create a workspace in Atagia."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    workspace_id: str | None = None
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatReplyRequest(BaseModel):
    """Main chat request payload."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    message_text: str
    attachments: list["AttachmentInput"] = Field(default_factory=list)
    message_occurred_at: str | None = None
    include_thinking: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    debug: bool = False

    @field_validator("message_occurred_at")
    @classmethod
    def validate_message_occurred_at(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized)
        return normalized


ArtifactType = Literal["url", "pdf", "image", "base64", "file", "pasted_text", "other"]
ArtifactSourceKind = Literal["host_embedded", "upload", "url", "base64", "pasted_text", "external_ref"]
ArtifactStatus = Literal["queued", "processing", "ready", "failed", "deleted", "purged"]
ArtifactRelationKind = Literal["attachment", "inline_ref", "citation", "imported_source"]
ArtifactChunkKind = Literal["ocr", "extracted", "parsed", "transcript", "summary"]


class AttachmentInput(BaseModel):
    """User-supplied attachment metadata and payload hints."""

    model_config = ConfigDict(extra="forbid")

    kind: ArtifactType
    content_text: str | None = None
    content_base64: str | None = None
    url: str | None = None
    source_ref: str | None = None
    filename: str | None = None
    title: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    privacy_level: int = Field(default=0, ge=0, le=3)
    preserve_verbatim: bool = False
    skip_raw_by_default: bool = True
    requires_explicit_request: bool = True
    size_bytes: int | None = Field(default=None, ge=0)
    page_count: int | None = Field(default=None, ge=0)

    @field_validator(
        "content_base64",
        "url",
        "source_ref",
        "filename",
        "title",
        "mime_type",
    )
    @classmethod
    def validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("content_text")
    @classmethod
    def validate_content_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        return normalized or None

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    @model_validator(mode="after")
    def validate_source_payload(self) -> "AttachmentInput":
        if not any(
            value is not None
            for value in (self.content_text, self.content_base64, self.url, self.source_ref)
        ):
            raise ValueError(
                "attachments require content_text, content_base64, url, or source_ref"
            )
        return self


ChatReplyRequest.model_rebuild()


class SidecarContextRequest(BaseModel):
    """Request payload for sidecar context retrieval."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    message_text: str
    assistant_mode_id: str | None = None
    workspace_id: str | None = None
    message_occurred_at: str | None = None
    attachments: list[AttachmentInput] = Field(default_factory=list)

    @field_validator("message_occurred_at")
    @classmethod
    def validate_message_occurred_at(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized)
        return normalized


class SidecarIngestMessageRequest(BaseModel):
    """Request payload for sidecar message ingestion."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    role: Literal["user", "assistant"]
    text: str
    assistant_mode_id: str | None = None
    workspace_id: str | None = None
    occurred_at: str | None = None
    attachments: list[AttachmentInput] = Field(default_factory=list)

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized)
        return normalized


class SidecarAddResponseRequest(BaseModel):
    """Request payload for persisting a host-generated assistant response."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    text: str
    occurred_at: str | None = None

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized)
        return normalized


class FlushRequest(BaseModel):
    """Request payload for waiting on pending sidecar background work."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)


class ArtifactRecord(BaseModel):
    """Canonical artifact record returned by the API and library mode."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    artifact_type: ArtifactType
    source_kind: ArtifactSourceKind
    source_ref: str | None = None
    mime_type: str | None = None
    filename: str | None = None
    title: str | None = None
    content_hash: str | None = None
    size_bytes: int | None = None
    page_count: int | None = None
    status: ArtifactStatus
    privacy_level: int = Field(ge=0, le=3)
    preserve_verbatim: bool = False
    skip_raw_by_default: bool = True
    requires_explicit_request: bool = True
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    summary_text: str | None = None
    index_text: str | None = None
    created_at: str
    updated_at: str
    deleted_at: str | None = None

    @field_validator("created_at", "updated_at", "deleted_at")
    @classmethod
    def validate_optional_timestamps(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return normalized

    @field_validator("metadata_json")
    @classmethod
    def validate_metadata_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)


class ArtifactChunkRecord(BaseModel):
    """Canonical artifact chunk record returned by the API and library mode."""

    model_config = ConfigDict(extra="forbid")

    id: str
    artifact_id: str
    user_id: str
    chunk_index: int = Field(ge=0)
    source_start_offset: int | None = Field(default=None, ge=0)
    source_end_offset: int | None = Field(default=None, ge=0)
    text: str
    token_count: int = Field(ge=0)
    kind: ArtifactChunkKind
    created_at: str
    updated_at: str

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamps(cls, value: str) -> str:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            raise ValueError("timestamps must be non-empty")
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return normalized


class ArtifactLinkRecord(BaseModel):
    """Canonical artifact link between a message and an artifact."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    message_id: str
    artifact_id: str
    relation_kind: ArtifactRelationKind
    ordinal: int = Field(ge=0)
    created_at: str

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            raise ValueError("created_at must be non-empty")
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return normalized


class ChatReplyResponse(BaseModel):
    """Response payload for the chat endpoint."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    request_message_id: str
    response_message_id: str
    reply_text: str
    retrieval_event_id: str | None = None
    debug: dict[str, Any] | None = None


class ChatResult(BaseModel):
    """Reusable full chat result for HTTP and library callers."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    request_message_id: str
    response_message_id: str
    response_text: str
    retrieval_event_id: str | None = None
    composed_context: ComposedContext | None = None
    detected_needs: list[str] = Field(default_factory=list)
    memories_used: list[dict[str, Any]] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


class MemorySummary(BaseModel):
    """Compact selected-memory view returned by library mode retrieval."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    text: str
    object_type: str
    score: float
    scope: str


class ContextResult(BaseModel):
    """Library-mode retrieval result ready for an external LLM call."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    memories: list[MemorySummary] = Field(default_factory=list)
    contract: dict[str, dict[str, Any]] = Field(default_factory=dict)
    detected_needs: list[str] = Field(default_factory=list)
    stage_timings: dict[str, float] = Field(default_factory=dict)
    from_cache: bool = False
    staleness: float = 1.0
    next_refresh_strategy: Literal["cache", "sync"] = "sync"
    cache_age_seconds: float | None = None
    cache_source: Literal["sync", "cache_hit"] | None = None
    need_detection_skipped: bool = False


class SidecarMutationResponse(BaseModel):
    """Small acknowledgement for sidecar write operations."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True


class FlushResponse(BaseModel):
    """Response payload for a background-work flush request."""

    model_config = ConfigDict(extra="forbid")

    completed: bool


class ConversationActivityStats(BaseModel):
    """Materialized conversation activity snapshot."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    conversation_id: str
    workspace_id: str | None = None
    assistant_mode_id: str
    timezone: str
    first_message_at: str | None = None
    last_message_at: str | None = None
    last_user_message_at: str | None = None
    message_count: int
    user_message_count: int
    assistant_message_count: int
    retrieval_count: int
    active_day_count: int
    recent_1d_message_count: int
    recent_7d_message_count: int
    recent_30d_message_count: int
    weekday_histogram_json: list[int] = Field(default_factory=list)
    hour_histogram_json: list[int] = Field(default_factory=list)
    hour_of_week_histogram_json: list[int] = Field(default_factory=list)
    return_interval_histogram_json: list[int] = Field(default_factory=list)
    avg_return_interval_minutes: float | None = None
    median_return_interval_minutes: float | None = None
    p90_return_interval_minutes: float | None = None
    main_thread_score: float
    likely_soon_score: float
    return_habit_confidence: float
    schedule_pattern_kind: str
    activity_version: int
    updated_at: str


class ActivitySnapshotResponse(BaseModel):
    """Activity listing returned by library and HTTP routes."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    as_of: str
    filters: dict[str, Any] = Field(default_factory=dict)
    conversations: list[ConversationActivityStats] = Field(default_factory=list)
    conversation_count: int


class ConversationWarmupRequest(BaseModel):
    """Warm-up payload for one conversation."""

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None
    max_messages: int = Field(default=12, ge=1, le=100)
    as_of: str | None = None

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return normalized


class UserWarmupRequest(BaseModel):
    """Warm-up payload for a ranked set of conversations."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=3, ge=1, le=20)
    workspace_id: str | None = None
    assistant_mode_id: str | None = None
    as_of: str | None = None
    lead_time_minutes: int | None = Field(default=None, ge=0, le=24 * 60)
    total_message_budget: int = Field(default=24, ge=1, le=200)
    per_conversation_message_budget: int = Field(default=12, ge=1, le=100)

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return normalized


class WarmupConversationResponse(BaseModel):
    """Warm-up result for one conversation."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    conversation_id: str
    as_of: str
    recent_window_key: str
    recent_message_count: int
    recent_message_ids: list[str] = Field(default_factory=list)
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    cached_context_key: str | None = None
    cached_context_available: bool = False
    stats: ConversationActivityStats | None = None
    warmup_errors: list[str] = Field(default_factory=list)


class WarmupRecommendedConversationsResponse(BaseModel):
    """Warm-up result for a hot-conversation batch."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    as_of: str
    requested_limit: int
    total_message_budget: int
    per_conversation_message_budget: int
    workspace_id: str | None = None
    assistant_mode_id: str | None = None
    hot_conversations: list[ConversationActivityStats] = Field(default_factory=list)
    warmed_conversations: list[WarmupConversationResponse] = Field(default_factory=list)
    warmed_conversation_count: int
    warmed_message_count: int


class VerbatimPinCreateRequest(BaseModel):
    """Request payload for creating a verbatim pin."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    scope: MemoryScope
    target_kind: VerbatimPinTargetKind
    target_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    assistant_mode_id: str | None = None
    canonical_text: str | None = None
    index_text: str | None = None
    target_span_start: int | None = Field(default=None, ge=0)
    target_span_end: int | None = Field(default=None, ge=0)
    privacy_level: int = Field(default=0, ge=0, le=3)
    reason: str | None = None
    created_by: str | None = None
    expires_at: str | None = None
    payload_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "target_id",
        "workspace_id",
        "conversation_id",
        "assistant_mode_id",
        "canonical_text",
        "index_text",
        "reason",
        "created_by",
    )
    @classmethod
    def validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return normalized

    @field_validator("target_span_end")
    @classmethod
    def validate_span_end(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return int(value)

    @field_validator("payload_json")
    @classmethod
    def validate_payload_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    @field_validator("target_span_start")
    @classmethod
    def validate_span_start(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return int(value)

    @field_validator("target_span_start", "target_span_end")
    @classmethod
    def validate_span_values(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("target span offsets must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_span_order(self) -> "VerbatimPinCreateRequest":
        if (
            self.target_span_start is not None
            and self.target_span_end is not None
            and self.target_span_end < self.target_span_start
        ):
            raise ValueError("target_span_end must be greater than or equal to target_span_start")
        return self


class VerbatimPinUpdateRequest(BaseModel):
    """Request payload for updating a verbatim pin."""

    model_config = ConfigDict(extra="forbid")

    canonical_text: str | None = None
    index_text: str | None = None
    target_span_start: int | None = Field(default=None, ge=0)
    target_span_end: int | None = Field(default=None, ge=0)
    privacy_level: int | None = Field(default=None, ge=0, le=3)
    status: VerbatimPinStatus | None = None
    reason: str | None = None
    expires_at: str | None = None
    payload_json: dict[str, Any] | None = None

    @field_validator("canonical_text", "index_text", "reason")
    @classmethod
    def validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return normalized

    @field_validator("payload_json")
    @classmethod
    def validate_payload_json(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return dict(value)

    @field_validator("target_span_start", "target_span_end")
    @classmethod
    def validate_span_values(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("target span offsets must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_span_order(self) -> "VerbatimPinUpdateRequest":
        if (
            self.target_span_start is not None
            and self.target_span_end is not None
            and self.target_span_end < self.target_span_start
        ):
            raise ValueError("target_span_end must be greater than or equal to target_span_start")
        return self


class VerbatimPinRecord(BaseModel):
    """Canonical verbatim pin record returned by the API and library mode."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    assistant_mode_id: str | None = None
    scope: MemoryScope
    target_kind: VerbatimPinTargetKind
    target_id: str
    target_span_start: int | None = None
    target_span_end: int | None = None
    canonical_text: str
    index_text: str
    privacy_level: int = Field(ge=0, le=3)
    status: VerbatimPinStatus
    reason: str | None = None
    created_by: str
    created_at: str
    updated_at: str
    expires_at: str | None = None
    deleted_at: str | None = None
    payload_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "id",
        "user_id",
        "workspace_id",
        "conversation_id",
        "assistant_mode_id",
        "target_id",
        "canonical_text",
        "index_text",
        "reason",
        "created_by",
        "created_at",
        "updated_at",
        "expires_at",
        "deleted_at",
    )
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("created_at", "updated_at", "expires_at", "deleted_at")
    @classmethod
    def validate_timestamps(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return normalized

    @field_validator("payload_json")
    @classmethod
    def validate_payload_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    @field_validator("target_span_start", "target_span_end")
    @classmethod
    def validate_span_values(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("target span offsets must be non-negative")
        return int(value)

    @model_validator(mode="after")
    def validate_span_order(self) -> "VerbatimPinRecord":
        if (
            self.target_span_start is not None
            and self.target_span_end is not None
            and self.target_span_end < self.target_span_start
        ):
            raise ValueError("target_span_end must be greater than or equal to target_span_start")
        return self


class MemoryFeedbackType(str, Enum):
    """API-level feedback values accepted by the memory feedback route."""

    USED = "used"
    USEFUL = "useful"
    IRRELEVANT = "irrelevant"
    INTRUSIVE = "intrusive"
    STALE = "stale"
    WRONG_SCOPE = "wrong_scope"
    CORRECTED_BY_USER = "corrected_by_user"
    CONFIRMED_BY_USER = "confirmed_by_user"


class MemoryFeedbackRequest(BaseModel):
    """User feedback on a retrieved memory."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    retrieval_event_id: str
    memory_id: str
    feedback_type: MemoryFeedbackType
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdminMetricsComputeRequest(BaseModel):
    """Request payload for on-demand metric computation."""

    model_config = ConfigDict(extra="forbid")

    time_bucket: str
    user_id: str | None = None
    assistant_mode_id: str | None = None
    metrics: list[MetricName] = Field(min_length=1)


class AdminEmbeddingBackfillRequest(BaseModel):
    """Request payload for admin-triggered embedding backfill."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=100, ge=1)
    delay_ms: int = Field(default=0, ge=0)
    user_id: str | None = None
