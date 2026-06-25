"""Background worker job schemas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atagia.models.schemas_memory import (
    ConfirmationStrategy,
    IngestOrigin,
    MemoryPrivacyMode,
    OperationalProfileSnapshot,
    resolve_confirmation_strategy,
)


class JobType(str, Enum):
    EXTRACT_MEMORY_CANDIDATES = "extract_memory_candidates"
    PROJECT_CONTRACT = "project_contract"
    REVISE_BELIEFS = "revise_beliefs"
    COMPACT_SUMMARIES = "compact_summaries"
    SYNC_GRAPH = "sync_graph"
    RUN_EVALUATION = "run_evaluation"
    REFRESH_INITIAL_CONTEXT_PACKAGE = "refresh_initial_context_package"


class JobRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"
    CANCELLED = "cancelled"


class WorkerControlMode(str, Enum):
    ACTIVE = "active"
    PAUSE_NEW_JOBS = "pause_new_jobs"
    DRAIN_AND_PAUSE = "drain_and_pause"
    HARD_PAUSE = "hard_pause"


class CompactionJobKind(str, Enum):
    CONVERSATION_CHUNK = "conversation_chunk"
    WORKSPACE_ROLLUP = "workspace_rollup"
    EPISODE = "episode"
    THEMATIC_PROFILE = "thematic_profile"


class InitialContextPackageRefreshReason(str, Enum):
    MESSAGE_WRITE = "message_write"
    MEMORY_EXTRACTION = "memory_extraction"
    CONTRACT_PROJECTION = "contract_projection"
    SUMMARY_COMPACTION = "summary_compaction"
    LIFECYCLE_CHANGE = "lifecycle_change"
    BACKFILL = "backfill"
    ADMIN_REBUILD = "admin_rebuild"
    COORDINATE_CHANGE = "coordinate_change"


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
    operational_profile: OperationalProfileSnapshot | None = None


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
    deferred: int = 0
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
    user_persona_id: str | None = None
    platform_id: str = "default"
    character_id: str | None = None
    active_presence_id: str | None = None
    active_presence_kind: str = "unknown"
    active_presence_display_name: str | None = None
    source_presence_id: str | None = None
    source_presence_kind: str = "unknown"
    source_presence_display_name: str | None = None
    active_space_id: str | None = None
    active_space_boundary_mode: str = "focus"
    active_space_display_name: str | None = None
    active_mind_id: str | None = None
    source_mind_id: str | None = None
    active_mind_display_name: str | None = None
    mind_topology: str = "unimind"
    active_embodiment_id: str | None = None
    active_embodiment_display_name: str | None = None
    cross_embodiment_mode: str = "direct_if_same_body"
    active_realm_id: str | None = None
    active_realm_display_name: str | None = None
    cross_realm_mode: str = "none"
    mode: str | None = None
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    temporary: bool = False
    temporary_ttl_seconds: int | None = None
    purge_on_close: bool = False
    valid_to: str | None = None
    isolated_mode: bool = False
    ingest_origin: IngestOrigin = IngestOrigin.LIVE_TURN
    confirmation_strategy: ConfirmationStrategy | None = None
    memory_privacy_mode: MemoryPrivacyMode = MemoryPrivacyMode.BALANCED
    privacy_enforcement: Literal["enforce", "audit_only", "off"] = "enforce"
    authenticated_user_privilege_level: str | None = None
    authenticated_user_is_atagia_master: bool = False

    @model_validator(mode="before")
    @classmethod
    def resolve_confirmation_strategy_default(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("confirmation_strategy") is None:
            normalized["confirmation_strategy"] = resolve_confirmation_strategy(
                ingest_origin=normalized.get("ingest_origin"),
                confirmation_strategy=None,
            )
        return normalized


class GraphProjectionChunkPayload(BaseModel):
    """One source-message chunk for graph projection."""

    model_config = ConfigDict(extra="forbid")

    text: str
    chunk_index: int = Field(default=1, ge=1)
    chunk_count: int = Field(default=1, ge=1)
    chunking_strategy: str | None = None
    level1_failure_reason: str | None = None
    level1_attempts: int = Field(default=0, ge=0)
    source_memory_ids: list[str] = Field(default_factory=list)


class GraphProjectionJobPayload(MessageJobPayload):
    """Payload used by the SQLite graph projection worker."""

    source_memory_ids: list[str] = Field(default_factory=list)
    chunks: list[GraphProjectionChunkPayload] = Field(min_length=1)


EXTRACT_STREAM_NAME = "atagia:extract"
CONTRACT_STREAM_NAME = "atagia:contract"
REVISE_STREAM_NAME = "atagia:revise"
COMPACT_STREAM_NAME = "atagia:compact"
GRAPH_STREAM_NAME = "atagia:graph"
EVALUATION_STREAM_NAME = "atagia:evaluate"
INITIAL_CONTEXT_PACKAGE_STREAM_NAME = "atagia:initial_context_package"
WORKER_GROUP_NAME = "atagia-workers"


class RevisionJobPayload(BaseModel):
    """Payload used by the revision worker."""

    model_config = ConfigDict(extra="forbid")

    belief_id: str = ""
    claim_key: str = Field(min_length=1)
    claim_value: str
    claim_key_already_validated: bool = False
    evidence_memory_ids: list[str] = Field(default_factory=list)
    source_message_id: str
    user_id: str
    assistant_mode_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    user_persona_id: str | None = None
    platform_id: str = "default"
    character_id: str | None = None
    active_mind_id: str | None = None
    source_mind_id: str | None = None
    mind_topology: str = "unimind"
    active_embodiment_id: str | None = None
    cross_embodiment_mode: str = "direct_if_same_body"
    active_realm_id: str | None = None
    cross_realm_mode: str = "none"
    mode: str | None = None
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True
    temporary: bool = False
    temporary_ttl_seconds: int | None = None
    purge_on_close: bool = False
    valid_to: str | None = None
    sensitivity: str = "unknown"
    platform_locked: bool = False
    platform_id_lock: str | None = None
    scope_canonical: str | None = None
    scope: str
    isolated_mode: bool = False


class CompactionJobPayload(BaseModel):
    """Payload used by the compaction worker."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    user_persona_id: str | None = None
    platform_id: str = "default"
    character_id: str | None = None
    mode: str | None = None
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True
    temporary: bool = False
    temporary_ttl_seconds: int | None = None
    purge_on_close: bool = False
    valid_to: str | None = None
    privacy_enforcement: Literal["enforce", "audit_only", "off"] = "enforce"
    job_kind: CompactionJobKind


class InitialContextPackageRefreshJobPayload(BaseModel):
    """Payload used by prepared initial-context package refresh jobs."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    conversation_id: str | None = None
    package_kind: Literal["all", "baseline", "conversation"] = "all"
    retrieval_profile_id: str | None = None
    reason: InitialContextPackageRefreshReason
    source_message_ids: list[str] = Field(default_factory=list)
    privacy_enforcement: Literal["enforce", "audit_only", "off"] = "enforce"


class EvaluationJobPayload(BaseModel):
    """Payload used by the evaluation worker."""

    model_config = ConfigDict(extra="forbid")

    time_bucket: str
    user_id: str | None = None
    assistant_mode_id: str | None = None
    user_persona_id: str | None = None
    platform_id: str = "default"
    character_id: str | None = None
    mode: str | None = None
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True
    metrics: list[str] = Field(min_length=1)
