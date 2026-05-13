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
            "low": "low",
            "medium": "medium",
            "high": "high",
        },
        default_thinking_level="minimal",
    ),
    "openai/gpt-5.5": ModelProfile(
        thinking_level_map={
            "none": "none",
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
        },
        default_thinking_level="none",
    ),
    "anthropic/claude-opus-4-7": ModelProfile(
        thinking_level_map={
            "none": None,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": "max",
        },
        default_thinking_level="none",
    ),
    "anthropic/claude-haiku-4-5": ModelProfile(
        thinking_level_map={
            "none": None,
            "minimal": 1024,
            "low": 1024,
            "medium": 4096,
            "high": 16384,
        },
        default_thinking_level="none",
    ),
    "anthropic/claude-sonnet-4-6": ModelProfile(
        thinking_level_map={
            "none": None,
            "minimal": 1024,
            "low": 1024,
            "medium": 4096,
            "high": -1,
        },
        default_thinking_level="none",
    ),
    "google/gemini-3-flash-preview": ModelProfile(
        thinking_level_map={
            "none": "MINIMAL",
            "minimal": "MINIMAL",
            "low": "LOW",
            "medium": "MEDIUM",
            "high": "HIGH",
        },
        default_thinking_level="minimal",
    ),
    "google/gemini-3.1-flash-lite": ModelProfile(
        thinking_level_map={
            "none": "MINIMAL",
            "minimal": "MINIMAL",
            "low": "LOW",
            "medium": "MEDIUM",
            "high": "HIGH",
        },
        default_thinking_level="minimal",
    ),
    "openrouter/google/gemini-3.1-flash-lite": ModelProfile(
        thinking_level_map={
            "none": "minimal",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
        },
        default_thinking_level="minimal",
        extra_body={"reasoning": {}},
    ),
    "openrouter/deepseek/deepseek-v4-flash": ModelProfile(
        thinking_level_map={
            "none": "none",
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
        },
        default_thinking_level="none",
        extra_body={"reasoning": {}},
    ),
    "openrouter/openai/gpt-5.5": ModelProfile(
        thinking_level_map={
            "none": "none",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
        },
        default_thinking_level="none",
        extra_body={"reasoning": {}},
    ),
    "openrouter/z-ai/glm-4.6": ModelProfile(),
    "openrouter/x-ai/grok-4.1-fast": ModelProfile(),
    "openrouter/x-ai/grok-4-fast": ModelProfile(),
}
