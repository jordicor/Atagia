"""Schemas for adaptive context cache artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atagia.models.schemas_api import MemorySummary
from atagia.models.schemas_memory import ComposedContext


class ContextCacheEntry(BaseModel):
    """Stable conversation-scoped cache entry for prompt-ready context."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    cache_key: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    assistant_mode_id: str = Field(min_length=1)
    policy_prompt_hash: str = Field(min_length=1)
    workspace_id: str | None = None
    composed_context: ComposedContext
    contract: dict[str, dict[str, Any]] = Field(default_factory=dict)
    memory_summaries: list[MemorySummary] = Field(default_factory=list)
    detected_needs: list[str] = Field(default_factory=list)
    source_retrieval_plan: dict[str, Any] = Field(default_factory=dict)
    selected_memory_ids: list[str] = Field(default_factory=list)
    cached_at: str = Field(min_length=1)
    last_retrieval_message_seq: int = Field(ge=0)
    last_user_message_text: str = Field(min_length=1)
    source: Literal["sync"] = "sync"

    @field_validator("policy_prompt_hash")
    @classmethod
    def validate_policy_prompt_hash(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("policy_prompt_hash must be non-empty")
        return normalized

    @field_validator("detected_needs")
    @classmethod
    def validate_detected_needs(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        if len(normalized) != len(set(normalized)):
            raise ValueError("detected_needs values must be unique")
        return normalized

    @field_validator("selected_memory_ids")
    @classmethod
    def validate_selected_memory_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        if len(normalized) != len(set(normalized)):
            raise ValueError("selected_memory_ids values must be unique")
        return normalized
