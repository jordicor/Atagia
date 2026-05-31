"""Tests for prepared initial-context package schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atagia.models.schemas_initial_context_package import (
    INITIAL_CONTEXT_PACKAGE_KEY_HASH_PREFIX,
    InitialContextPackageBlocks,
    InitialContextPackageKey,
    InitialContextPackageKind,
    InitialContextPackageProfileItem,
    InitialContextPackageRecord,
    InitialContextPackageSourceFingerprint,
    initial_context_package_key_hash,
)


def _baseline_key(**overrides: object) -> InitialContextPackageKey:
    payload = {
        "version": 1,
        "package_kind": InitialContextPackageKind.BASELINE,
        "user_id": "usr_1",
        "conversation_id": None,
        "retrieval_profile_id": "default",
        "subject_json": {"platform_id": "aurvek", "character_id": "core"},
        "policy_json": {"effective_policy_hash": "policy-a"},
        "coordinate_json": {"space_id": "space-main", "mind_topology": "unimind"},
        "operational_json": {"profile": "normal"},
    }
    payload.update(overrides)
    return InitialContextPackageKey.model_validate(payload)


def test_key_hash_is_canonical_and_prefixed() -> None:
    first = _baseline_key(
        subject_json={"platform_id": "aurvek", "character_id": "core"},
        coordinate_json={"space_id": "space-main", "mind_topology": "unimind"},
    )
    second = _baseline_key(
        coordinate_json={"mind_topology": "unimind", "space_id": "space-main"},
        subject_json={"character_id": "core", "platform_id": "aurvek"},
    )

    first_hash = initial_context_package_key_hash(first)
    assert first_hash == initial_context_package_key_hash(second)
    assert first_hash.startswith(INITIAL_CONTEXT_PACKAGE_KEY_HASH_PREFIX)


def test_key_validates_baseline_and_conversation_shape() -> None:
    with pytest.raises(ValidationError, match="baseline packages must not include"):
        _baseline_key(conversation_id="cnv_1")

    with pytest.raises(ValidationError, match="conversation packages require"):
        InitialContextPackageKey(
            version=1,
            package_kind=InitialContextPackageKind.CONVERSATION,
            user_id="usr_1",
            retrieval_profile_id="default",
        )

    conversation_key = InitialContextPackageKey(
        version=1,
        package_kind=InitialContextPackageKind.CONVERSATION,
        user_id="usr_1",
        conversation_id="cnv_1",
        retrieval_profile_id="default",
    )
    assert conversation_key.conversation_id == "cnv_1"


def test_profile_items_require_source_refs() -> None:
    item = InitialContextPackageProfileItem(
        item_id="profile_style_es",
        text="The user normally writes in Spanish.",
        reason_category="communication_profile",
        source_refs=[{"source_kind": "communication_profile", "profile_id": "ucp_1"}],
    )
    assert item.source_refs[0]["profile_id"] == "ucp_1"

    with pytest.raises(ValidationError, match="source_refs"):
        InitialContextPackageProfileItem(
            item_id="profile_without_source",
            text="The user prefers short answers.",
            reason_category="preference",
            source_refs=[],
        )


def test_record_fields_must_match_key() -> None:
    key = _baseline_key()
    record_payload = {
        "id": "icp_test",
        "package_key_hash": initial_context_package_key_hash(key),
        "package_kind": InitialContextPackageKind.BASELINE,
        "version": 1,
        "user_id": "usr_1",
        "conversation_id": None,
        "retrieval_profile_id": "default",
        "key_json": key,
        "policy_signature_json": {},
        "coordinate_signature_json": {},
        "source_fingerprint_json": InitialContextPackageSourceFingerprint(
            source_fingerprint_hash="fp-a",
            source_markers_json={"memory_objects_max_updated_at": "2026-06-08T00:00:00+00:00"},
        ),
        "blocks_json": InitialContextPackageBlocks(
            prepared_memory_profile_block="Known stable profile.",
            source_counts={"profile_items": 1},
        ),
        "source_refs_json": {},
        "diagnostics_json": {},
        "build_status": "active",
        "created_at": "2026-06-08T00:00:00+00:00",
        "updated_at": "2026-06-08T00:00:00+00:00",
    }

    record = InitialContextPackageRecord.model_validate(record_payload)
    assert record.user_id == "usr_1"
    assert record.source_fingerprint_json.source_fingerprint_hash == "fp-a"

    mismatched = {**record_payload, "user_id": "usr_2"}
    with pytest.raises(ValidationError, match="record user_id must match"):
        InitialContextPackageRecord.model_validate(mismatched)
