"""Model profile metadata for provider-specific request tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Provider-specific metadata for a fully qualified Atagia model spec."""

    thinking_level_map: dict[str, str | int | None] = field(default_factory=dict)
    default_thinking_level: str | None = None
    extra_body: dict[str, Any] | None = None
    extra_kwargs: dict[str, Any] | None = None


MODEL_PROFILES: dict[str, ModelProfile] = {
    "openai/gpt-4o-mini": ModelProfile(),
    "openai/gpt-5-mini": ModelProfile(
        thinking_level_map={
            "none": "minimal",
            "minimal": "minimal",
            "medium": "medium",
            "high": "high",
        },
        default_thinking_level="minimal",
    ),
    "openai/gpt-5.5": ModelProfile(
        thinking_level_map={
            "none": "none",
            "minimal": "low",
            "medium": "medium",
            "high": "high",
        },
        default_thinking_level="none",
    ),
    "anthropic/claude-haiku-4-5": ModelProfile(
        thinking_level_map={
            "none": None,
            "minimal": 1024,
            "medium": 4096,
            "high": 16384,
        },
        default_thinking_level="none",
    ),
    "anthropic/claude-sonnet-4-6": ModelProfile(
        thinking_level_map={
            "none": None,
            "minimal": 1024,
            "medium": 4096,
            "high": -1,
        },
        default_thinking_level="none",
    ),
    "google/gemini-3-flash-preview": ModelProfile(
        thinking_level_map={
            "none": "MINIMAL",
            "minimal": "MINIMAL",
            "medium": "MEDIUM",
            "high": "HIGH",
        },
        default_thinking_level="minimal",
    ),
    "google/gemini-3.1-flash-lite-preview": ModelProfile(
        thinking_level_map={
            "none": "MINIMAL",
            "minimal": "MINIMAL",
            "medium": "MEDIUM",
            "high": "HIGH",
        },
        default_thinking_level="minimal",
    ),
    "openrouter/google/gemini-3.1-flash-lite-preview": ModelProfile(
        thinking_level_map={
            "none": "minimal",
            "minimal": "minimal",
            "medium": "medium",
            "high": "high",
        },
        default_thinking_level="minimal",
        extra_body={"reasoning": {}},
    ),
    "openrouter/deepseek/deepseek-v4-flash": ModelProfile(
        thinking_level_map={
            "none": "none",
            "minimal": "low",
            "medium": "medium",
            "high": "high",
        },
        default_thinking_level="none",
        extra_body={"reasoning": {}},
    ),
}
