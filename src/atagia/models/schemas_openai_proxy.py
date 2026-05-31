"""OpenAI-compatible request and response models for the Atagia memory proxy."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atagia.models.schemas_memory import ResponseMode


class OpenAIProxyMessage(BaseModel):
    """OpenAI-compatible chat message accepted by the proxy."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class OpenAIChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat-completion request used by the memory proxy."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[OpenAIProxyMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any | None = None
    stream_options: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    user: str | None = None
    # Atagia extension: optional per-request latency/quality mode. Kept
    # OpenAI-schema-tolerant (the model already allows extra fields); an
    # explicit field gives enum validation when callers set it directly.
    response_mode: ResponseMode | None = None


class OpenAIModelObject(BaseModel):
    """OpenAI-compatible model-list item."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "atagia"


class OpenAIModelList(BaseModel):
    """OpenAI-compatible model-list response."""

    object: str = "list"
    data: list[OpenAIModelObject]
