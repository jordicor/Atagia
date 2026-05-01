"""Tests for structured high-risk memory policy helpers."""

from __future__ import annotations

from atagia.memory.high_risk_policy import (
    HighRiskDisclosureAction,
    disclosure_action,
    requires_confirmation,
)
from atagia.models.schemas_memory import MemoryCategory


def test_structured_high_risk_categories_require_confirmation() -> None:
    assert requires_confirmation(
        memory_category=MemoryCategory.PIN_OR_PASSWORD,
        privacy_level=1,
    )
    assert requires_confirmation(
        memory_category=MemoryCategory.MEDICATION,
        privacy_level=1,
    )
    assert requires_confirmation(
        memory_category=MemoryCategory.UNKNOWN,
        privacy_level=3,
    )


def test_routine_memory_does_not_require_high_risk_confirmation() -> None:
    assert not requires_confirmation(
        memory_category=MemoryCategory.UNKNOWN,
        privacy_level=1,
    )


def test_secret_literals_are_withheld_in_ordinary_chat() -> None:
    assert (
        disclosure_action(
            memory_category=MemoryCategory.PIN_OR_PASSWORD,
            privacy_level=1,
            preserve_verbatim=False,
        )
        == HighRiskDisclosureAction.WITHHOLD_SECRET_LITERAL
    )
    assert (
        disclosure_action(
            memory_category=MemoryCategory.UNKNOWN,
            privacy_level=3,
            preserve_verbatim=True,
        )
        == HighRiskDisclosureAction.WITHHOLD_SECRET_LITERAL
    )


def test_sensitive_non_secret_details_can_be_answered_when_authorized() -> None:
    assert (
        disclosure_action(
            memory_category=MemoryCategory.MEDICATION,
            privacy_level=2,
            preserve_verbatim=False,
        )
        == HighRiskDisclosureAction.ANSWER_IF_AUTHORIZED
    )
