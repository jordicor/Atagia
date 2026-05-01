"""Reusable helpers for host-application Atagia integrations."""

from atagia.integrations.message_projection import message_to_text
from atagia.integrations.prompt_injection import (
    ATAGIA_CONTEXT_FOOTER,
    ATAGIA_CONTEXT_HEADER,
    ContextInjectionDecision,
    append_context_to_prompt,
    build_injection_decision,
    context_messages_for_provider,
    extract_context_message_id,
    extract_context_system_prompt,
)
from atagia.integrations.sidecar_bridge import (
    DEFAULT_ASSISTANT_MODE,
    DEFAULT_TIMEOUT_SECONDS,
    AtagiaClientProtocol,
    ClientFactory,
    ConfigLoader,
    SidecarBridge,
    SidecarBridgeConfig,
)

__all__ = [
    "ATAGIA_CONTEXT_FOOTER",
    "ATAGIA_CONTEXT_HEADER",
    "DEFAULT_ASSISTANT_MODE",
    "DEFAULT_TIMEOUT_SECONDS",
    "AtagiaClientProtocol",
    "ClientFactory",
    "ConfigLoader",
    "ContextInjectionDecision",
    "SidecarBridge",
    "SidecarBridgeConfig",
    "append_context_to_prompt",
    "build_injection_decision",
    "context_messages_for_provider",
    "extract_context_message_id",
    "extract_context_system_prompt",
    "message_to_text",
]
