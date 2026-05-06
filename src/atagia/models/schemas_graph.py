"""Schemas for the lightweight SQLite graph projection."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atagia.models.schemas_memory import IntimacyBoundary, MemoryScope


class GraphEntityResolution(str, Enum):
    """How the projection model resolved an entity candidate."""

    NEW = "new"
    EXISTING = "existing"
    AMBIGUOUS = "ambiguous"


class GraphEntityStatus(str, Enum):
    """Lifecycle states for projected graph entities and aliases."""

    ACTIVE = "active"
    REVIEW_REQUIRED = "review_required"
    MERGED = "merged"
    ARCHIVED = "archived"
    DELETED = "deleted"


class GraphRelationshipStatus(str, Enum):
    """Lifecycle states for projected relationship edges."""

    ACTIVE = "active"
    REVIEW_REQUIRED = "review_required"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"
    ARCHIVED = "archived"
    DELETED = "deleted"


class GraphSourceKind(str, Enum):
    """Canonical source kinds graph rows may point back to."""

    MESSAGE = "message"
    ARTIFACT = "artifact"
    ARTIFACT_CHUNK = "artifact_chunk"
    MEMORY_OBJECT = "memory_object"


class GraphRelationshipDirection(str, Enum):
    """Directionality for graph relationship rows."""

    DIRECTED = "directed"
    SYMMETRIC = "symmetric"


GraphEntityType = Literal[
    "person",
    "organization",
    "place",
    "project",
    "document",
    "work",
    "event",
    "concept",
    "product",
    "other",
]


class GraphEntityCandidate(BaseModel):
    """One entity candidate emitted by the graph projection model."""

    model_config = ConfigDict(extra="ignore")

    local_id: str = Field(min_length=1)
    entity_type: GraphEntityType
    display_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    resolution: GraphEntityResolution = GraphEntityResolution.NEW
    existing_entity_id: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: GraphEntityStatus = GraphEntityStatus.ACTIVE
    evidence_quote: str | None = None
    privacy_level: int = Field(default=0, ge=0, le=3)
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("local_id", "display_name", mode="before")
    @classmethod
    def normalize_required_text(cls, value: Any) -> str:
        normalized = " ".join(str(value).split())
        if not normalized:
            raise ValueError("value must be non-empty")
        return normalized

    @field_validator("aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, values: Any) -> list[str]:
        if values is None:
            return []
        if not isinstance(values, list):
            return []
        aliases: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(str(value).split())
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            aliases.append(normalized)
        return aliases

    @model_validator(mode="after")
    def validate_existing_resolution(self) -> "GraphEntityCandidate":
        if self.resolution is GraphEntityResolution.EXISTING and not self.existing_entity_id:
            raise ValueError("existing resolution requires existing_entity_id")
        return self


class GraphRelationshipCandidate(BaseModel):
    """One relationship candidate emitted by the graph projection model."""

    model_config = ConfigDict(extra="ignore")

    source_local_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    target_local_id: str | None = None
    target_value: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    direction: GraphRelationshipDirection = GraphRelationshipDirection.DIRECTED
    scope: MemoryScope = MemoryScope.CONVERSATION
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: GraphRelationshipStatus = GraphRelationshipStatus.ACTIVE
    valid_from_iso: str | None = None
    valid_to_iso: str | None = None
    evidence_quote: str = Field(min_length=1)
    privacy_level: int = Field(default=0, ge=0, le=3)
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supersedes_local_relationship_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_local_id", "predicate", mode="before")
    @classmethod
    def normalize_required_text(cls, value: Any) -> str:
        normalized = " ".join(str(value).split())
        if not normalized:
            raise ValueError("value must be non-empty")
        return normalized

    @field_validator("target_local_id", "valid_from_iso", "valid_to_iso", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split())
        return normalized or None

    @field_validator("evidence_quote", mode="before")
    @classmethod
    def normalize_evidence_quote(cls, value: Any) -> str:
        normalized = " ".join(str(value).split())
        if not normalized:
            raise ValueError("evidence_quote must be non-empty")
        return normalized

    @field_validator("valid_from_iso", "valid_to_iso")
    @classmethod
    def validate_temporal_iso(cls, value: str | None) -> str | None:
        if value is None:
            return None
        datetime.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def validate_target_and_temporal_bounds(self) -> "GraphRelationshipCandidate":
        if self.target_local_id is None and self.target_value is None:
            raise ValueError("relationship requires target_local_id or target_value")
        if self.valid_from_iso and self.valid_to_iso:
            if datetime.fromisoformat(self.valid_from_iso) > datetime.fromisoformat(self.valid_to_iso):
                raise ValueError("valid_from_iso must be <= valid_to_iso")
        return self


class GraphProjectionResult(BaseModel):
    """Structured output returned by the graph projector."""

    model_config = ConfigDict(extra="ignore")

    entities: list[GraphEntityCandidate] = Field(default_factory=list)
    relationships: list[GraphRelationshipCandidate] = Field(default_factory=list)
    nothing_durable: bool = False
    skipped_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_root_aliases(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"entities": data, "relationships": [], "nothing_durable": False}
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "entities" not in normalized:
            for field_name in ("entity_candidates", "nodes"):
                if field_name in normalized:
                    normalized["entities"] = normalized.pop(field_name)
                    break
        if "relationships" not in normalized:
            for field_name in ("relationship_candidates", "edges"):
                if field_name in normalized:
                    normalized["relationships"] = normalized.pop(field_name)
                    break
        return normalized

    @model_validator(mode="after")
    def validate_nothing_durable_consistency(self) -> "GraphProjectionResult":
        if self.nothing_durable and (self.entities or self.relationships):
            raise ValueError("nothing_durable=true but graph projection lists are non-empty")
        return self
