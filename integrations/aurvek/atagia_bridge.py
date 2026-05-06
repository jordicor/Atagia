"""Copyable Aurvek-style bridge built on Atagia's generic sidecar helper.

Aurvek's live private bridge originally carried this logic inline. This
reference version keeps the public integration shape small and delegates the
transport, fail-open behavior, and Atagia client calls to
``atagia.integrations.SidecarBridge``.
"""

from __future__ import annotations

from typing import Any

from atagia.integrations import (
    AURVEK_PLATFORM_ID,
    DEFAULT_MODE,
    DEFAULT_TIMEOUT_SECONDS,
    ConfirmationStrategy,
    IngestOrigin,
    MemoryPrivacyMode,
    SidecarBridge,
    SidecarBridgeConfig,
    aurvek_conversation_id,
    aurvek_message_id,
    aurvek_prompt_character_id,
    aurvek_user_id,
)

AtagiaBridgeConfig = SidecarBridgeConfig
AtagiaBridge = SidecarBridge

_default_bridge: AtagiaBridge | None = None


def get_atagia_bridge() -> AtagiaBridge:
    """Return the process-wide Atagia bridge singleton."""
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = AtagiaBridge(AtagiaBridgeConfig.from_env())
    return _default_bridge


async def get_context_for_turn(
    user_id: int | str,
    conversation_id: int | str,
    message_text: str,
    *,
    occurred_at: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    message_id: int | str | None = None,
    source_seq: int | str | None = None,
    user_persona_id: str | None = None,
    platform_id: str | None = AURVEK_PLATFORM_ID,
    character_id: str | None = None,
    prompt_id: int | str | None = None,
    mode: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
    incognito: bool | None = None,
    ingest_origin: IngestOrigin | str | None = None,
    confirmation_strategy: ConfirmationStrategy | str | None = None,
    memory_privacy_mode: MemoryPrivacyMode | str | None = None,
) -> Any | None:
    """Return Atagia context for one Aurvek turn.

    This copyable example uses the canonical runtime bridge. It maps Aurvek's
    host IDs into Atagia namespaced IDs before calling the package API.
    """
    return await get_atagia_bridge().get_context_for_turn(
        _aurvek_user_id(user_id),
        _aurvek_conversation_id(conversation_id),
        message_text,
        occurred_at=occurred_at,
        attachments=attachments,
        user_persona_id=user_persona_id,
        platform_id=platform_id or AURVEK_PLATFORM_ID,
        character_id=_resolve_character_id(character_id, prompt_id),
        mode=mode,
        operational_profile=operational_profile,
        operational_signals=operational_signals,
        incognito=incognito,
        message_id=_resolve_message_id(message_id),
        source_seq=_resolve_source_seq(source_seq),
        ingest_origin=ingest_origin,
        confirmation_strategy=confirmation_strategy,
        memory_privacy_mode=memory_privacy_mode,
    )


async def record_assistant_response(
    user_id: int | str,
    conversation_id: int | str,
    response_text: str,
    *,
    occurred_at: str | None = None,
    message_id: int | str | None = None,
    source_seq: int | str | None = None,
    user_persona_id: str | None = None,
    platform_id: str | None = AURVEK_PLATFORM_ID,
    character_id: str | None = None,
    prompt_id: int | str | None = None,
    mode: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
    incognito: bool | None = None,
    ingest_origin: IngestOrigin | str | None = None,
    confirmation_strategy: ConfirmationStrategy | str | None = None,
    memory_privacy_mode: MemoryPrivacyMode | str | None = None,
) -> bool:
    """Persist an Aurvek assistant response in Atagia."""
    return await get_atagia_bridge().record_assistant_response(
        _aurvek_user_id(user_id),
        _aurvek_conversation_id(conversation_id),
        response_text,
        occurred_at=occurred_at,
        user_persona_id=user_persona_id,
        platform_id=platform_id or AURVEK_PLATFORM_ID,
        character_id=_resolve_character_id(character_id, prompt_id),
        mode=mode,
        operational_profile=operational_profile,
        operational_signals=operational_signals,
        incognito=incognito,
        message_id=_resolve_message_id(message_id),
        source_seq=_resolve_source_seq(source_seq),
        ingest_origin=ingest_origin,
        confirmation_strategy=confirmation_strategy,
        memory_privacy_mode=memory_privacy_mode,
    )


async def ingest_message(
    user_id: int | str,
    conversation_id: int | str,
    role: str,
    text: str,
    *,
    occurred_at: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    message_id: int | str | None = None,
    source_seq: int | str | None = None,
    user_persona_id: str | None = None,
    platform_id: str | None = AURVEK_PLATFORM_ID,
    character_id: str | None = None,
    prompt_id: int | str | None = None,
    mode: str | None = None,
    operational_profile: str | None = None,
    operational_signals: dict[str, Any] | None = None,
    incognito: bool | None = None,
    ingest_origin: IngestOrigin | str | None = None,
    confirmation_strategy: ConfirmationStrategy | str | None = None,
    memory_privacy_mode: MemoryPrivacyMode | str | None = None,
) -> bool:
    """Persist an Aurvek historical or sidecar message in Atagia.

    Pass ``ingest_origin="backfill"`` for historical sync/import paths so
    sensitive candidates are routed to admin review instead of user prompts.
    Pass ``memory_privacy_mode="trusted_private"`` only after explicit user
    opt-in; Atagia then treats sensitive/imported candidates as trusted writes.
    """
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    return await get_atagia_bridge().ingest_message(
        _aurvek_user_id(user_id),
        _aurvek_conversation_id(conversation_id),
        role,
        text,
        occurred_at=occurred_at,
        attachments=attachments,
        user_persona_id=user_persona_id,
        platform_id=platform_id or AURVEK_PLATFORM_ID,
        character_id=_resolve_character_id(character_id, prompt_id),
        mode=mode,
        operational_profile=operational_profile,
        operational_signals=operational_signals,
        incognito=incognito,
        message_id=_resolve_message_id(message_id),
        source_seq=_resolve_source_seq(source_seq),
        ingest_origin=ingest_origin,
        confirmation_strategy=confirmation_strategy,
        memory_privacy_mode=memory_privacy_mode,
    )


async def list_pending_memory_confirmations(
    user_id: int | str,
    **filters: Any,
) -> Any | None:
    """Return safe pending confirmations for Aurvek Settings -> Memory."""
    return await get_atagia_bridge().list_pending_memory_confirmations(
        _aurvek_user_id(user_id),
        **filters,
    )


async def confirm_pending_memory(
    user_id: int | str,
    memory_id: int | str,
) -> Any | None:
    """Confirm one pending memory."""
    return await get_atagia_bridge().confirm_pending_memory(
        _aurvek_user_id(user_id),
        str(memory_id),
    )


async def decline_pending_memory(
    user_id: int | str,
    memory_id: int | str,
) -> Any | None:
    """Decline one pending memory."""
    return await get_atagia_bridge().decline_pending_memory(
        _aurvek_user_id(user_id),
        str(memory_id),
    )


async def list_review_required_memories(**filters: Any) -> Any | None:
    """Return admin review-required memories."""
    return await get_atagia_bridge().list_review_required_memories(**filters)


async def archive_review_required_memory(
    user_id: int | str,
    memory_id: int | str,
) -> Any | None:
    """Archive one review-required memory."""
    return await get_atagia_bridge().archive_review_required_memory(
        _aurvek_user_id(user_id),
        str(memory_id),
    )


async def delete_review_required_memory(
    user_id: int | str,
    memory_id: int | str,
) -> Any | None:
    """Delete one review-required memory."""
    return await get_atagia_bridge().delete_review_required_memory(
        _aurvek_user_id(user_id),
        str(memory_id),
    )


async def pause_new_jobs(*, reason: str | None = None) -> Any | None:
    """Ask Atagia to store messages but stop queuing new memory jobs."""
    return await get_atagia_bridge().pause_new_jobs(reason=reason)


async def drain_and_pause(
    *,
    reason: str | None = None,
    timeout_seconds: float | None = None,
) -> Any | None:
    """Ask Atagia to drain already queued work, then stay paused."""
    return await get_atagia_bridge().drain_and_pause(
        reason=reason,
        timeout_seconds=timeout_seconds,
    )


async def hard_pause(*, reason: str | None = None) -> Any | None:
    """Ask Atagia workers to stop claiming further jobs."""
    return await get_atagia_bridge().hard_pause(reason=reason)


async def resume_processing(*, reason: str | None = None) -> Any | None:
    """Ask Atagia to resume normal background processing."""
    return await get_atagia_bridge().resume_processing(reason=reason)


async def close_atagia_bridge() -> None:
    """Close the process-wide bridge client if it has been initialized."""
    if _default_bridge is not None:
        await _default_bridge.close()


async def reset_atagia_bridge() -> None:
    """Close and drop the process-wide bridge so fresh env config is loaded."""
    global _default_bridge
    if _default_bridge is not None:
        await _default_bridge.close()
    _default_bridge = None


def _aurvek_user_id(value: int | str) -> str:
    text = str(value).strip()
    return text if text.startswith("aurvek:user:") else aurvek_user_id(text)


def _aurvek_conversation_id(value: int | str) -> str:
    text = str(value).strip()
    return (
        text
        if text.startswith("aurvek:conv:")
        else aurvek_conversation_id(text)
    )


def _resolve_message_id(value: int | str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if text.startswith("aurvek:msg:") else aurvek_message_id(text)


def _resolve_source_seq(value: int | str | None) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 1:
        raise ValueError("source_seq must be a positive integer")
    return resolved


def _resolve_character_id(
    character_id: str | None,
    prompt_id: int | str | None,
) -> str | None:
    if character_id is not None and character_id.strip():
        return character_id.strip()
    if prompt_id is None:
        return None
    return aurvek_prompt_character_id(prompt_id)


__all__ = [
    "AtagiaBridge",
    "AtagiaBridgeConfig",
    "ConfirmationStrategy",
    "DEFAULT_MODE",
    "DEFAULT_TIMEOUT_SECONDS",
    "IngestOrigin",
    "archive_review_required_memory",
    "close_atagia_bridge",
    "confirm_pending_memory",
    "decline_pending_memory",
    "delete_review_required_memory",
    "drain_and_pause",
    "get_atagia_bridge",
    "get_context_for_turn",
    "hard_pause",
    "ingest_message",
    "list_pending_memory_confirmations",
    "list_review_required_memories",
    "pause_new_jobs",
    "record_assistant_response",
    "reset_atagia_bridge",
    "resume_processing",
]
