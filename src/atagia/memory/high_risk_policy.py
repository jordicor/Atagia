"""Structured high-risk memory policy helpers."""

from __future__ import annotations

from enum import Enum

from atagia.models.schemas_memory import MemoryCategory


class HighRiskDisclosureAction(str, Enum):
    """Response-layer action for a retrieved high-risk memory."""

    ANSWER_IF_AUTHORIZED = "answer_if_authorized"
    WITHHOLD_SECRET_LITERAL = "withhold_secret_literal"


CONFIRMATION_REQUIRED_MEMORY_CATEGORIES = frozenset(
    {
        MemoryCategory.PIN_OR_PASSWORD,
        MemoryCategory.MEDICATION,
        MemoryCategory.FINANCIAL,
        MemoryCategory.DATE_OF_BIRTH,
    }
)

SECRET_LITERAL_MEMORY_CATEGORIES = frozenset({MemoryCategory.PIN_OR_PASSWORD})

HIGH_RISK_CHAT_POLICY_INSTRUCTION = (
    "High-risk memory policy: retrieval permission is not the same as raw "
    "disclosure permission. If retrieved context contains a high-risk secret "
    "literal, do not reveal the raw value in an ordinary chat answer, even when "
    "the memory was retrieved. This non-disclosure rule applies to "
    "`memory_category: pin_or_password` memories and to requests whose target is "
    "clearly an authentication credential, personal PIN, password, payment-card "
    "secret, recovery phrase, or production passphrase. Do not generalize this "
    "rule to every remembered code or numeric value. Building, delivery, room, "
    "event, appointment, or other logistics codes are not automatically covered "
    "by this raw-secret rule when their memory category is not "
    "`pin_or_password`; they are governed by the active user's scope, privacy "
    "ceiling, status, and explicit disclosure conditions. When withholding a "
    "raw secret literal, say that you cannot disclose that secret in chat and "
    "direct the user to a host-managed secure reveal or verification flow. You "
    "may acknowledge that relevant protected information exists when that is "
    "useful, but do not quote or transform the secret. For other sensitive "
    "personal details, including medication, financial logistics, addresses, and "
    "dates of birth, answer only when the current authenticated user, assistant "
    "mode, scope, privacy ceiling, and any explicit disclosure condition permit "
    "it; otherwise withhold. For all high-risk material, do not substitute "
    "nearby facts or infer missing values; if older and newer permitted values "
    "conflict, answer only with the current active value."
)


def requires_confirmation(
    *,
    memory_category: MemoryCategory,
    privacy_level: int,
) -> bool:
    """Return whether storing a user-supplied item starts behind consent."""
    return (
        privacy_level >= 3
        or memory_category in CONFIRMATION_REQUIRED_MEMORY_CATEGORIES
    )


def disclosure_action(
    *,
    memory_category: MemoryCategory,
    privacy_level: int,
    preserve_verbatim: bool,
) -> HighRiskDisclosureAction:
    """Return the ordinary-chat disclosure action for structured memory metadata."""
    if memory_category in SECRET_LITERAL_MEMORY_CATEGORIES:
        return HighRiskDisclosureAction.WITHHOLD_SECRET_LITERAL
    if preserve_verbatim and privacy_level >= 3:
        return HighRiskDisclosureAction.WITHHOLD_SECRET_LITERAL
    return HighRiskDisclosureAction.ANSWER_IF_AUTHORIZED
