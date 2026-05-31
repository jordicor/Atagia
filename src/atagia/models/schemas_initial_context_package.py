"""Schemas for prepared initial context package artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atagia.core.canonical import canonical_json_hash

INITIAL_CONTEXT_PACKAGE_KEY_HASH_PREFIX = "icp:v1:"


class InitialContextPackageKind(str, Enum):
    """Durable package families for prepared context."""

    BASELINE = "baseline"
    CONVERSATION = "conversation"


class InitialContextPackageBuildStatus(str, Enum):
    """Persistence status for prepared package rows."""

    ACTIVE = "active"
    STALE = "stale"
    FAILED = "failed"
    DELETED = "deleted"


class InitialContextPackageKey(BaseModel):
    """Deterministic package identity before hashing."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    package_kind: InitialContextPackageKind
    user_id: str = Field(min_length=1)
    conversation_id: str | None = None
    retrieval_profile_id: str = Field(min_length=1)
    subject_json: dict[str, Any] = Field(default_factory=dict)
    policy_json: dict[str, Any] = Field(default_factory=dict)
    coordinate_json: dict[str, Any] = Field(default_factory=dict)
    operational_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("user_id", "conversation_id", "retrieval_profile_id")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier fields must be non-empty")
        return normalized

    @model_validator(mode="after")
    def validate_conversation_shape(self) -> "InitialContextPackageKey":
        if (
            self.package_kind == InitialContextPackageKind.CONVERSATION
            and self.conversation_id is None
        ):
            raise ValueError("conversation packages require conversation_id")
        if (
            self.package_kind == InitialContextPackageKind.BASELINE
            and self.conversation_id is not None
        ):
            raise ValueError("baseline packages must not include conversation_id")
        return self


class InitialContextPackagePolicySignature(BaseModel):
    """Policy and authority markers used to validate package visibility."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    effective_policy_hash: str | None = None
    policy_prompt_hash: str | None = None
    privacy_enforcement: str | None = None
    authority_json: dict[str, Any] = Field(default_factory=dict)
    markers_json: dict[str, Any] = Field(default_factory=dict)


class InitialContextPackageCoordinateSignature(BaseModel):
    """Coordinate markers used to validate package visibility."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    coordinate_signature_hash: str | None = None
    complete: bool = True
    markers_json: dict[str, Any] = Field(default_factory=dict)


class InitialContextPackageSourceFingerprint(BaseModel):
    """Durable source markers used to detect stale package rows."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    source_fingerprint_hash: str | None = None
    source_markers_json: dict[str, Any] = Field(default_factory=dict)


class InitialContextPackageProfileItem(BaseModel):
    """One governed factual item in the prepared memory profile."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    reason_category: str = Field(min_length=1)
    source_refs: list[dict[str, Any]] = Field(min_length=1)
    scope_json: dict[str, Any] = Field(default_factory=dict)
    coordinate_visibility_json: dict[str, Any] = Field(default_factory=dict)
    freshness_json: dict[str, Any] = Field(default_factory=dict)
    status: Literal["current", "historical", "superseded", "ambiguous"] = "current"
    salience: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("item_id", "text", "reason_category")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("profile item text fields must be non-empty")
        return normalized


class InitialContextPackageBlocks(BaseModel):
    """Structured blocks that can be rendered into the prompt path."""

    model_config = ConfigDict(extra="forbid")

    contract_block: str = ""
    curated_orientation_block: str = ""
    prepared_memory_profile_block: str = ""
    current_state_block: str = ""
    coordinate_context_block: str = ""
    conversation_summary_block: str = ""
    working_topic_block: str = ""
    recent_verbatim_seed: list[dict[str, Any]] = Field(default_factory=list)
    empty_markers: dict[str, bool] = Field(default_factory=dict)
    source_counts: dict[str, int] = Field(default_factory=dict)
    curated_items: list[InitialContextPackageProfileItem] = Field(default_factory=list)
    profile_items: list[InitialContextPackageProfileItem] = Field(default_factory=list)

    @field_validator("source_counts")
    @classmethod
    def validate_source_counts(cls, value: dict[str, int]) -> dict[str, int]:
        for key, count in value.items():
            if not key.strip():
                raise ValueError("source_counts keys must be non-empty")
            if int(count) < 0:
                raise ValueError("source_counts values must be non-negative")
        return value


class InitialContextPackageDiagnostics(BaseModel):
    """Non-sensitive diagnostics for package reads and builds."""

    model_config = ConfigDict(extra="forbid")

    package_tokens_estimate: int = Field(default=0, ge=0)
    source_counts: dict[str, int] = Field(default_factory=dict)
    selected_profile_items: int = Field(default=0, ge=0)
    dropped_profile_items: int = Field(default=0, ge=0)
    selected_curated_items: int = Field(default=0, ge=0)
    dropped_curated_items: int = Field(default=0, ge=0)
    refresh_job_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


class InitialContextPackageRecord(BaseModel):
    """Decoded durable package row."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    package_key_hash: str = Field(min_length=1)
    package_kind: InitialContextPackageKind
    version: int = Field(ge=1)
    user_id: str = Field(min_length=1)
    conversation_id: str | None = None
    retrieval_profile_id: str = Field(min_length=1)
    key_json: InitialContextPackageKey
    policy_signature_json: InitialContextPackagePolicySignature
    coordinate_signature_json: InitialContextPackageCoordinateSignature
    source_fingerprint_json: InitialContextPackageSourceFingerprint
    blocks_json: InitialContextPackageBlocks
    source_refs_json: dict[str, Any] = Field(default_factory=dict)
    diagnostics_json: InitialContextPackageDiagnostics = Field(
        default_factory=InitialContextPackageDiagnostics
    )
    build_status: InitialContextPackageBuildStatus = (
        InitialContextPackageBuildStatus.ACTIVE
    )
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    valid_until: str | None = None

    @field_validator(
        "id",
        "package_key_hash",
        "user_id",
        "conversation_id",
        "retrieval_profile_id",
        "created_at",
        "updated_at",
        "valid_until",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("record text fields must be non-empty")
        return normalized

    @model_validator(mode="after")
    def validate_record_shape(self) -> "InitialContextPackageRecord":
        if self.package_kind != self.key_json.package_kind:
            raise ValueError("record package_kind must match key_json")
        if self.user_id != self.key_json.user_id:
            raise ValueError("record user_id must match key_json")
        if self.conversation_id != self.key_json.conversation_id:
            raise ValueError("record conversation_id must match key_json")
        if self.retrieval_profile_id != self.key_json.retrieval_profile_id:
            raise ValueError("record retrieval_profile_id must match key_json")
        return self


class InitialContextPackageReadResult(BaseModel):
    """Repository read result with non-sensitive diagnostics."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["hit", "miss", "stale", "deleted", "unavailable"]
    package: InitialContextPackageRecord | None = None
    fallback_reason: str | None = None


def initial_context_package_key_hash(
    key: InitialContextPackageKey | Mapping[str, Any],
) -> str:
    """Return the stable key hash used by package repository reads."""
    payload = key.model_dump(mode="json") if isinstance(key, BaseModel) else dict(key)
    return INITIAL_CONTEXT_PACKAGE_KEY_HASH_PREFIX + canonical_json_hash(payload)
