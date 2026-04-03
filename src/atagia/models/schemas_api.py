"""API request and response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atagia.core.timestamps import normalize_optional_timestamp
from atagia.models.schemas_evaluation import MetricName
from atagia.models.schemas_memory import ComposedContext


class ChatMessageInput(BaseModel):
    """Minimal input payload for a chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateConversationRequest(BaseModel):
    """Create a conversation in Atagia."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    assistant_mode_id: str
    workspace_id: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateWorkspaceRequest(BaseModel):
    """Create a workspace in Atagia."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatReplyRequest(BaseModel):
    """Main chat request payload."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    message_text: str
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
