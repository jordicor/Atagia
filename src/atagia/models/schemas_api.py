"""API request and response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atagia.core.timestamps import normalize_optional_timestamp
from atagia.models.schemas_evaluation import MetricName
from atagia.models.schemas_jobs import WorkerControlMode
from atagia.models.schemas_memory import (
    ComposedContext,
    ConfirmationStrategy,
    IngestOrigin,
    IntimacyBoundary,
    MemoryCategory,
    MemoryPrivacyMode,
    MemoryScope,
    MemorySensitivity,
    OperationalSignals,
    TopicWorkingSetTrace,
    VerbatimPinStatus,
    VerbatimPinTargetKind,
    resolve_confirmation_strategy,
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
    cross_chat_memory: bool = True
    temporary: bool = False
    temporary_ttl_seconds: int | None = Field(default=None, gt=0)
    purge_on_close: bool | None = None
    # Namespace redesign identity fields. They are accepted alongside the
    # legacy fields during the additive phase; Phase 11 will drop the
    # legacy ones. ``platform_id`` becomes required at the service /
    # proxy / sidecar boundary in Phase 4 wiring; library callers may
    # still omit it for now.
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    mode: str | None = None
    incognito: bool | None = None


class CloseConversationRequest(BaseModel):
    """Close a conversation without deleting data unless explicitly requested."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    platform_id: str
    user_persona_id: str | None = None
    character_id: str | None = None
    incognito: bool | None = None
    purge: bool | None = None
    confirmation: str | None = None


class ConversationLifecycleRequest(BaseModel):
    """Conversation lifecycle request scoped by user."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    platform_id: str
    user_persona_id: str | None = None
    character_id: str | None = None
    incognito: bool | None = None


class MemoryPreferencesResponse(BaseModel):
    """Resolved user-level memory sharing preferences."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    remember_across_chats: bool = True
    remember_across_devices: bool = True
    memory_privacy_mode: MemoryPrivacyMode = MemoryPrivacyMode.BALANCED


class UpdateMemoryPreferencesRequest(BaseModel):
    """Partial update for user-level memory sharing preferences."""

    model_config = ConfigDict(extra="forbid")

    remember_across_chats: bool | None = None
    remember_across_devices: bool | None = None
    memory_privacy_mode: MemoryPrivacyMode | None = None


class ConversationIncognitoRequest(BaseModel):
    """Set the reversible per-conversation incognito flag."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    incognito: bool
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None


class SaveFromIncognitoRequest(BaseModel):
    """Explicit rescue request for memories from an incognito conversation."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    mode: str | None = None


class IncognitoReviewMessage(BaseModel):
    """One source message proposed for explicit incognito rescue review."""

    model_config = ConfigDict(extra="forbid")

    message_id: str
    role: Literal["user", "assistant"]
    seq: int
    text: str
    occurred_at: str | None = None
    content_kind: str = "text"
    policy_reason: str = "normal"
    skip_by_default: bool = False


class SaveFromIncognitoResponse(BaseModel):
    """Review manifest for an explicit incognito rescue request."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    conversation_id: str
    status: Literal["review_required"]
    review_policy: Literal["non_incognito"]
    source_message_count: int = Field(ge=0)
    source_messages: list[IncognitoReviewMessage] = Field(default_factory=list)
    suggested_memory_count: int = Field(ge=0)
    suggested_memories: list[dict[str, Any]] = Field(default_factory=list)
    writes_performed: bool = False


class DeleteConversationRequest(BaseModel):
    """Hard-delete a conversation and its conversation-scoped data."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    platform_id: str
    user_persona_id: str | None = None
    character_id: str | None = None
    incognito: bool | None = None
    confirmation: str


class EditMemoryRequest(BaseModel):
    """Edit active evidence memory text."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    conversation_id: str
    platform_id: str
    user_persona_id: str | None = None
    character_id: str | None = None
    incognito: bool | None = None
    canonical_text: str = Field(min_length=1)


class DeleteMemoryRequest(BaseModel):
    """Archive or hard-delete a memory object."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    conversation_id: str
    platform_id: str
    user_persona_id: str | None = None
    character_id: str | None = None
    incognito: bool | None = None
    hard: bool = False
    confirmation: str | None = None


class EraseUserDataRequest(BaseModel):
    """Erase all Atagia data for a user."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    confirmation: str


class DeletionReport(BaseModel):
    """Count-only report for a conversation or memory deletion."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = None
    memory_id: str | None = None
    deleted_memories: int = 0
    deleted_messages: int = 0
    deleted_summaries: int = 0
    deleted_artifacts: int = 0
    tombstone_id: str | None = None
    already_deleted: bool = False


class ErasureReport(BaseModel):
    """Count-only report for a user erasure."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    deleted_memories: int = 0
    deleted_conversations: int = 0
    deleted_artifacts: int = 0
    tombstone_id: str | None = None
    already_erased: bool = False


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
    operational_profile: str | None = Field(default=None, max_length=64)
    operational_signals: OperationalSignals | None = None
    cross_chat_memory: bool = True
    # Namespace redesign per-turn identity / privacy hints. They override
    # the conversation defaults for this turn only (e.g. setting
    # ``incognito=True`` on a single reply). ``mode`` is a retrieval
    # profile hint and never participates in scope SQL.
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    mode: str | None = None
    incognito: bool | None = None

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
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
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


def _resolve_sidecar_confirmation_strategy_default(data: object) -> object:
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    if normalized.get("confirmation_strategy") is None:
        normalized["confirmation_strategy"] = resolve_confirmation_strategy(
            ingest_origin=normalized.get("ingest_origin"),
            confirmation_strategy=None,
        )
    return normalized


class SidecarContextRequest(BaseModel):
    """Request payload for sidecar context retrieval."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    message_text: str
    message_id: str | None = None
    source_seq: int | None = Field(default=None, ge=1)
    assistant_mode_id: str | None = None
    workspace_id: str | None = None
    message_occurred_at: str | None = None
    attachments: list[AttachmentInput] = Field(default_factory=list)
    operational_profile: str | None = Field(default=None, max_length=64)
    operational_signals: OperationalSignals | None = None
    cross_chat_memory: bool = True
    # Namespace redesign identity fields. ``platform_id`` becomes
    # required for sidecar callers in Phase 4 wiring; until then it is
    # optional so existing integrations keep working.
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    mode: str | None = None
    incognito: bool | None = None
    ingest_origin: IngestOrigin = IngestOrigin.LIVE_TURN
    confirmation_strategy: ConfirmationStrategy | None = None
    memory_privacy_mode: MemoryPrivacyMode | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_confirmation_strategy_default(cls, data: object) -> object:
        return _resolve_sidecar_confirmation_strategy_default(data)

    @field_validator("message_occurred_at")
    @classmethod
    def validate_message_occurred_at(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized)
        return normalized

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class SidecarIngestMessageRequest(BaseModel):
    """Request payload for sidecar message ingestion."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    message_id: str | None = None
    source_seq: int | None = Field(default=None, ge=1)
    role: Literal["user", "assistant"]
    text: str
    assistant_mode_id: str | None = None
    workspace_id: str | None = None
    occurred_at: str | None = None
    attachments: list[AttachmentInput] = Field(default_factory=list)
    operational_profile: str | None = Field(default=None, max_length=64)
    operational_signals: OperationalSignals | None = None
    cross_chat_memory: bool = True
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    mode: str | None = None
    incognito: bool | None = None
    ingest_origin: IngestOrigin = IngestOrigin.LIVE_TURN
    confirmation_strategy: ConfirmationStrategy | None = None
    memory_privacy_mode: MemoryPrivacyMode | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_confirmation_strategy_default(cls, data: object) -> object:
        return _resolve_sidecar_confirmation_strategy_default(data)

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized)
        return normalized

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class SidecarAddResponseRequest(BaseModel):
    """Request payload for persisting a host-generated assistant response."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    message_id: str | None = None
    source_seq: int | None = Field(default=None, ge=1)
    text: str
    occurred_at: str | None = None
    operational_profile: str | None = Field(default=None, max_length=64)
    operational_signals: OperationalSignals | None = None
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    mode: str | None = None
    incognito: bool | None = None
    ingest_origin: IngestOrigin = IngestOrigin.LIVE_TURN
    confirmation_strategy: ConfirmationStrategy | None = None
    memory_privacy_mode: MemoryPrivacyMode | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_confirmation_strategy_default(cls, data: object) -> object:
        return _resolve_sidecar_confirmation_strategy_default(data)

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: str | None) -> str | None:
        normalized = normalize_optional_timestamp(value)
        if normalized is None:
            return None
        datetime.fromisoformat(normalized)
        return normalized

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class PendingMemoryConfirmationRecord(BaseModel):
    """Safe pending-confirmation item for host user interfaces."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    user_id: str
    conversation_id: str
    category: MemoryCategory
    label: str
    created_at: str
    asked_at: str | None = None
    confirmation_asked_once: bool = False
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    mode: str | None = None
    incognito_snapshot: bool = False
    intended_scope: str | None = None
    intended_sensitivity: MemorySensitivity = MemorySensitivity.UNKNOWN
    platform_locked: bool = False
    platform_id_lock: str | None = None
    policy_proven: bool = False
    memory_status: str | None = None


class PendingMemoryConfirmationListResponse(BaseModel):
    """Pending-confirmation collection response."""

    model_config = ConfigDict(extra="forbid")

    items: list[PendingMemoryConfirmationRecord]


class PendingMemoryConfirmationActionResponse(BaseModel):
    """Result of confirming or declining a pending memory."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    memory_id: str
    status: str


class AdminReviewMemoryRecord(BaseModel):
    """Admin-visible review-required memory item."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    user_id: str
    conversation_id: str | None = None
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    mode: str | None = None
    object_type: str
    category: MemoryCategory
    scope: str
    scope_canonical: str | None = None
    sensitivity: MemorySensitivity = MemorySensitivity.UNKNOWN
    privacy_level: int = Field(ge=0, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    canonical_text: str
    index_text: str | None = None
    review_reason: str | None = None
    ingest_origin: str | None = None
    confirmation_strategy: str | None = None
    memory_privacy_mode: str | None = None
    source_message_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class AdminReviewMemoryListResponse(BaseModel):
    """Admin review-required collection response."""

    model_config = ConfigDict(extra="forbid")

    items: list[AdminReviewMemoryRecord]


class AdminReviewActionResponse(BaseModel):
    """Result of an admin action on a review-required memory."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    memory_id: str
    status: str


class FlushRequest(BaseModel):
    """Request payload for waiting on pending sidecar background work."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    conversation_id: str | None = None
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)


class MemoryProcessingEstimate(BaseModel):
    """Rough remaining-time estimate for queued memory work."""

    model_config = ConfigDict(extra="forbid")

    estimated_remaining_seconds: float | None = Field(default=None, ge=0.0)
    estimate_range_seconds: list[float] | None = None
    confidence: Literal["none", "low", "medium", "high"] = "none"
    basis: Literal["none", "current_jobs", "historical_jobs"] = "none"

    @field_validator("estimate_range_seconds")
    @classmethod
    def validate_estimate_range(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("estimate_range_seconds must contain exactly two values")
        if value[0] < 0 or value[1] < 0:
            raise ValueError("estimate_range_seconds values must be non-negative")
        if value[1] < value[0]:
            raise ValueError("estimate_range_seconds upper bound must be >= lower bound")
        return value


class MemoryProcessingStatus(BaseModel):
    """Current durable-memory background processing status."""

    model_config = ConfigDict(extra="forbid")

    workers_enabled: bool
    processing: bool
    status: Literal["idle", "queued", "running", "retrying", "blocked", "degraded"]
    pending_source_messages: int = Field(ge=0)
    processed_source_messages: int = Field(ge=0)
    tracked_source_messages: int = Field(ge=0)
    pending_jobs: int = Field(ge=0)
    running_jobs: int = Field(ge=0)
    retrying_jobs: int = Field(ge=0)
    failed_jobs: int = Field(ge=0)
    dead_lettered_jobs: int = Field(ge=0)
    pending_jobs_by_type: dict[str, int] = Field(default_factory=dict)
    running_jobs_by_type: dict[str, int] = Field(default_factory=dict)
    oldest_pending_age_seconds: float | None = Field(default=None, ge=0.0)
    newest_job_queued_at: str | None = None
    estimate: MemoryProcessingEstimate = Field(default_factory=MemoryProcessingEstimate)
    global_queue_state: Literal["idle", "normal", "busy", "backlogged"] = "idle"
    global_pending_jobs: int = Field(default=0, ge=0)
    global_running_jobs: int = Field(default=0, ge=0)


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
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
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
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
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
    memory_processing: MemoryProcessingStatus | None = None
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
    memory_processing: MemoryProcessingStatus | None = None
    debug: dict[str, Any] | None = None


class MemorySummary(BaseModel):
    """Compact selected-memory view returned by library mode retrieval."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    text: str
    object_type: str
    score: float
    scope: str


class RecentTranscriptEntry(BaseModel):
    """One entry in the deterministic recent transcript returned to sidecar clients."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["message", "policy_placeholder", "omission"]
    role: Literal["user", "assistant"]
    text: str
    seq: int
    message_id: str
    token_estimate: int = Field(ge=0)
    content_kind: str = "text"
    policy_reason: str = "normal"
    omission_reason: Literal["token_budget", "policy"] | None = None


class RecentTranscriptOmission(BaseModel):
    """Structured metadata for a recent transcript message omitted from raw context."""

    model_config = ConfigDict(extra="forbid")

    reason: Literal["token_budget", "policy"]
    seq: int
    message_id: str
    role: Literal["user", "assistant"]
    token_estimate: int = Field(ge=0)
    content_kind: str = "text"
    policy_reason: str = "normal"


class RecentTranscriptTrace(BaseModel):
    """Trace metadata for sidecar recent transcript assembly."""

    model_config = ConfigDict(extra="forbid")

    budget_tokens: int = Field(ge=0)
    budget_used_tokens: int = Field(ge=0)
    overage_ratio: float = Field(ge=0.0)
    included_message_seqs: list[int] = Field(default_factory=list)
    omitted_message_seqs: list[int] = Field(default_factory=list)


class ContextResult(BaseModel):
    """Library-mode retrieval result ready for an external LLM call."""

    model_config = ConfigDict(extra="forbid")

    request_message_id: str | None = None
    system_prompt: str
    topic_working_set: TopicWorkingSetTrace = Field(default_factory=TopicWorkingSetTrace)
    topic_working_set_block: str = ""
    recent_transcript: list[RecentTranscriptEntry] = Field(default_factory=list)
    recent_transcript_omissions: list[RecentTranscriptOmission] = Field(default_factory=list)
    recent_transcript_trace: RecentTranscriptTrace | None = None
    assistant_guidance: list[str] = Field(default_factory=list)
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
    memory_processing: MemoryProcessingStatus | None = None


class SidecarMutationResponse(BaseModel):
    """Small acknowledgement for sidecar write operations."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    message_id: str | None = None
    seq: int | None = None
    source_seq: int | None = None
    idempotent_replay: bool = False


class FlushResponse(BaseModel):
    """Response payload for a background-work flush request."""

    model_config = ConfigDict(extra="forbid")

    completed: bool
    memory_processing: MemoryProcessingStatus | None = None


class WorkerControlRequest(BaseModel):
    """Request payload for the admin background-processing stop switch."""

    model_config = ConfigDict(extra="forbid")

    mode: WorkerControlMode
    reason: str | None = Field(default=None, max_length=500)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class WorkerControlResponse(BaseModel):
    """Current admin background-processing control state."""

    model_config = ConfigDict(extra="forbid")

    mode: WorkerControlMode
    reason: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None
    new_source_jobs_allowed: bool
    worker_claims_allowed: bool
    periodic_work_allowed: bool
    drain_completed: bool | None = None


class ConversationActivityStats(BaseModel):
    """Materialized conversation activity snapshot."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    conversation_id: str
    workspace_id: str | None = None
    assistant_mode_id: str
    # Namespace redesign identity / privacy fields. They are populated by
    # migration 0031 and by Phase 3 repository writes; older snapshots may
    # leave them as None / defaults.
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True
    effective_policy_hash: str | None = None
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
    platform_id: str
    user_persona_id: str | None = None
    character_id: str | None = None
    incognito: bool | None = None
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

    conversation_id: str
    limit: int = Field(default=3, ge=1, le=20)
    platform_id: str
    user_persona_id: str | None = None
    character_id: str | None = None
    incognito: bool = False
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
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    incognito: bool = False
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
    user_persona_id: str | None = None
    platform_id: str
    character_id: str | None = None
    incognito: bool | None = None
    canonical_text: str | None = None
    index_text: str | None = None
    target_span_start: int | None = Field(default=None, ge=0)
    target_span_end: int | None = Field(default=None, ge=0)
    privacy_level: int = Field(default=0, ge=0, le=3)
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = None
    created_by: str | None = None
    expires_at: str | None = None
    payload_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "target_id",
        "workspace_id",
        "conversation_id",
        "assistant_mode_id",
        "user_persona_id",
        "platform_id",
        "character_id",
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
    intimacy_boundary: IntimacyBoundary | None = None
    intimacy_boundary_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
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
    # Namespace redesign identity / sensitivity / platform-lock fields. They
    # are populated by migration 0031 and by Phase 4 pin writes; older pins
    # leave them as None / defaults.
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    sensitivity: MemorySensitivity = MemorySensitivity.UNKNOWN
    themes_json: list[str] = Field(default_factory=list)
    platform_locked: bool = False
    platform_id_lock: str | None = None
    scope_canonical: str | None = None
    incognito_snapshot: bool = False
    remember_across_chats_snapshot: bool = True
    remember_across_devices_snapshot: bool = True
    policy_snapshot_json: dict[str, Any] = Field(default_factory=dict)
    user_persona_key: str | None = None
    character_key: str | None = None
    conversation_key: str | None = None
    scope: MemoryScope
    target_kind: VerbatimPinTargetKind
    target_id: str
    target_span_start: int | None = None
    target_span_end: int | None = None
    canonical_text: str
    index_text: str
    privacy_level: int = Field(ge=0, le=3)
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
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
    conversation_id: str
    platform_id: str
    user_persona_id: str | None = None
    character_id: str | None = None
    incognito: bool | None = None
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
