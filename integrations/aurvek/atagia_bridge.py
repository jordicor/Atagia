"""Copyable Aurvek-style bridge built on Atagia's generic sidecar helper.

Aurvek's live private bridge originally carried this logic inline. This
reference version keeps the public integration shape small and delegates the
transport, fail-open behavior, and Atagia client calls to
``atagia.integrations.SidecarBridge``.
"""

from __future__ import annotations

from typing import Any

from atagia.integrations import (
    DEFAULT_ASSISTANT_MODE,
    DEFAULT_TIMEOUT_SECONDS,
    SidecarBridge,
    SidecarBridgeConfig,
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
) -> Any | None:
    """Return Atagia context for one Aurvek turn."""
    return await get_atagia_bridge().get_context_for_turn(
        user_id,
        conversation_id,
        message_text,
        occurred_at=occurred_at,
        attachments=attachments,
    )


async def record_assistant_response(
    user_id: int | str,
    conversation_id: int | str,
    response_text: str,
    *,
    occurred_at: str | None = None,
) -> bool:
    """Persist an Aurvek assistant response in Atagia."""
    return await get_atagia_bridge().record_assistant_response(
        user_id,
        conversation_id,
        response_text,
        occurred_at=occurred_at,
    )


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


__all__ = [
    "AtagiaBridge",
    "AtagiaBridgeConfig",
    "DEFAULT_ASSISTANT_MODE",
    "DEFAULT_TIMEOUT_SECONDS",
    "close_atagia_bridge",
    "get_atagia_bridge",
    "get_context_for_turn",
    "record_assistant_response",
    "reset_atagia_bridge",
]
