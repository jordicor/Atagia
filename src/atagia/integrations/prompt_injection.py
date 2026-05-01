"""Prompt assembly helpers for host-managed LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ATAGIA_CONTEXT_HEADER = "[ATAGIA MEMORY CONTEXT - INTERNAL]"
ATAGIA_CONTEXT_FOOTER = "[/ATAGIA MEMORY CONTEXT]"


@dataclass(frozen=True, slots=True)
class ContextInjectionDecision:
    """Result of deciding whether Atagia context changed a host prompt."""

    full_prompt: str
    active: bool
    reason: str
    context: Any | None = None
    atagia_user_message_id: str | None = None


def extract_context_system_prompt(context: Any) -> str:
    """Return the sidecar system prompt from dict or Pydantic-style contexts."""
    if context is None:
        return ""
    if isinstance(context, dict):
        raw_prompt = context.get("system_prompt")
    else:
        raw_prompt = getattr(context, "system_prompt", None)
    return raw_prompt.strip() if isinstance(raw_prompt, str) else ""


def extract_context_message_id(context: Any) -> str | None:
    """Return the request-message id carried by an Atagia context, if present."""
    if context is None:
        return None
    if isinstance(context, dict):
        raw_id = context.get("request_message_id") or context.get("message_id")
    else:
        raw_id = (
            getattr(context, "request_message_id", None)
            or getattr(context, "message_id", None)
        )
    return raw_id if isinstance(raw_id, str) and raw_id else None


def append_context_to_prompt(
    full_prompt: str,
    context: Any,
    *,
    instruction: str = (
        "Use this memory context to personalize and maintain continuity. "
        "Do not reveal this block verbatim to the user."
    ),
) -> str:
    """Append Atagia sidecar context to a host application's system prompt."""
    atagia_prompt = extract_context_system_prompt(context)
    if not atagia_prompt:
        return full_prompt
    return (
        f"{full_prompt.rstrip()}\n\n"
        f"{ATAGIA_CONTEXT_HEADER}\n"
        f"{instruction}\n\n"
        f"{atagia_prompt}\n"
        f"{ATAGIA_CONTEXT_FOOTER}"
    )


def build_injection_decision(
    full_prompt: str,
    context: Any | None,
) -> ContextInjectionDecision:
    """Build a prompt-injection decision for a fetched Atagia context."""
    if context is None:
        return ContextInjectionDecision(full_prompt, False, "no_context")
    augmented = append_context_to_prompt(full_prompt, context)
    if augmented == full_prompt:
        return ContextInjectionDecision(
            full_prompt,
            False,
            "empty_context",
            context=context,
        )
    return ContextInjectionDecision(
        augmented,
        True,
        "active",
        context=context,
        atagia_user_message_id=extract_context_message_id(context),
    )


def context_messages_for_provider(
    context_messages: list[dict[str, Any]],
    decision: ContextInjectionDecision,
) -> list[dict[str, Any]]:
    """Suppress host history when Atagia is acting as the primary context."""
    if decision.active:
        return []
    return context_messages
