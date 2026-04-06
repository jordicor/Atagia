"""Background worker job schemas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobType(str, Enum):
    EXTRACT_MEMORY_CANDIDATES = "extract_memory_candidates"
    PROJECT_CONTRACT = "project_contract"
    REVISE_BELIEFS = "revise_beliefs"
    COMPACT_SUMMARIES = "compact_summaries"
    SYNC_GRAPH = "sync_graph"
    RUN_EVALUATION = "run_evaluation"


class CompactionJobKind(str, Enum):
    CONVERSATION_CHUNK = "conversation_chunk"
    WORKSPACE_ROLLUP = "workspace_rollup"
    EPISODE = "episode"
    THEMATIC_PROFILE = "thematic_profile"


class JobEnvelope(BaseModel):
    """Stream- or queue-friendly job envelope."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    job_type: JobType
    user_id: str
    conversation_id: str | None = None
    message_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class StreamMessage(BaseModel):
    """Message consumed from a stream-like backend."""

    model_config = ConfigDict(extra="forbid")

    message_id: str
    payload: dict[str, Any]
    delivery_count: int = 1


@dataclass(frozen=True, slots=True)
class WorkerIterationResult:
    """Single poll result for a stream worker."""

    received: int = 0
    acked: int = 0
    failed: int = 0
    dead_lettered: int = 0


class MessageJobPayload(BaseModel):
    """Payload used by message-derived worker jobs."""

    model_config = ConfigDict(extra="forbid")

    message_id: str
    message_text: str
    message_occurred_at: str | None = None
    role: str
    assistant_mode_id: str
    workspace_id: str | None = None
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)


EXTRACT_STREAM_NAME = "atagia:extract"
CONTRACT_STREAM_NAME = "atagia:contract"
REVISE_STREAM_NAME = "atagia:revise"
COMPACT_STREAM_NAME = "atagia:compact"
EVALUATION_STREAM_NAME = "atagia:evaluate"
WORKER_GROUP_NAME = "atagia-workers"


class RevisionJobPayload(BaseModel):
    """Payload used by the revision worker."""

    model_config = ConfigDict(extra="forbid")

    belief_id: str = ""
    claim_key: str = Field(min_length=1)
    claim_value: str
    evidence_memory_ids: list[str] = Field(default_factory=list)
    source_message_id: str
    user_id: str
    assistant_mode_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    scope: str


class CompactionJobPayload(BaseModel):
    """Payload used by the compaction worker."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    job_kind: CompactionJobKind


class EvaluationJobPayload(BaseModel):
    """Payload used by the evaluation worker."""

    model_config = ConfigDict(extra="forbid")

    time_bucket: str
    user_id: str | None = None
    assistant_mode_id: str | None = None
    metrics: list[str] = Field(min_length=1)
