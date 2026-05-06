"""Reusable helpers for host-application Atagia integrations."""

from atagia.integrations.aurvek import (
    AURVEK_PLATFORM_ID,
    AurvekNamespace,
    aurvek_conversation_id,
    aurvek_message_id,
    aurvek_prompt_character_id,
    aurvek_user_id,
)
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
    DEFAULT_MODE,
    DEFAULT_TIMEOUT_SECONDS,
    AtagiaClientProtocol,
    ClientFactory,
    ConfigLoader,
    SidecarBridge,
    SidecarBridgeConfig,
    SidecarBridgeError,
)
from atagia.models.schemas_jobs import WorkerControlMode
from atagia.models.schemas_memory import (
    ConfirmationStrategy,
    IngestOrigin,
    MemoryPrivacyMode,
)

__all__ = [
    "ATAGIA_CONTEXT_FOOTER",
    "ATAGIA_CONTEXT_HEADER",
    "AURVEK_PLATFORM_ID",
    "DEFAULT_MODE",
    "DEFAULT_TIMEOUT_SECONDS",
    "AtagiaClientProtocol",
    "AurvekNamespace",
    "ClientFactory",
    "ConfigLoader",
    "ConfirmationStrategy",
    "ContextInjectionDecision",
    "SidecarBridge",
    "SidecarBridgeConfig",
    "SidecarBridgeError",
    "IngestOrigin",
    "MemoryPrivacyMode",
    "WorkerControlMode",
    "append_context_to_prompt",
    "aurvek_conversation_id",
    "aurvek_message_id",
    "aurvek_prompt_character_id",
    "aurvek_user_id",
    "build_injection_decision",
    "context_messages_for_provider",
    "extract_context_message_id",
    "extract_context_system_prompt",
    "message_to_text",
]
