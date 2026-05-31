"""Rendering tests for prepared initial-context package prompt blocks."""

from __future__ import annotations

from atagia.models.schemas_initial_context_package import (
    InitialContextPackageBlocks,
    InitialContextPackageKey,
    InitialContextPackageKind,
    InitialContextPackageProfileItem,
    InitialContextPackageRecord,
    InitialContextPackageSourceFingerprint,
    initial_context_package_key_hash,
)
from atagia.services.initial_context_package_builder import (
    INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
)
from atagia.services.initial_context_package_prompt import _render_packages


def _record() -> InitialContextPackageRecord:
    key = InitialContextPackageKey(
        version=INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
        package_kind=InitialContextPackageKind.BASELINE,
        user_id="usr_1",
        retrieval_profile_id="default",
    )
    curated = InitialContextPackageProfileItem(
        item_id="curated:1",
        text="The small trust repair conversation is pivotal context.",
        reason_category="relationship_orientation",
        source_refs=[{"source_kind": "memory_object", "memory_id": "mem_trust"}],
        status="historical",
        salience=0.9,
    )
    profile = InitialContextPackageProfileItem(
        item_id="memory:mem_profile",
        text="The user prefers direct Spanish replies.",
        reason_category="preference",
        source_refs=[{"source_kind": "memory_object", "memory_id": "mem_profile"}],
    )
    return InitialContextPackageRecord(
        id="icp_prompt_render",
        package_key_hash=initial_context_package_key_hash(key),
        package_kind=InitialContextPackageKind.BASELINE,
        version=INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
        user_id="usr_1",
        retrieval_profile_id="default",
        key_json=key,
        policy_signature_json={},
        coordinate_signature_json={},
        source_fingerprint_json=InitialContextPackageSourceFingerprint(
            source_fingerprint_hash="fingerprint-a",
        ),
        blocks_json=InitialContextPackageBlocks(
            curated_orientation_block="not used directly when curated_items exist",
            prepared_memory_profile_block="not used directly when profile_items exist",
            curated_items=[curated],
            profile_items=[profile],
            source_counts={"curated_items": 1, "profile_items": 1},
        ),
        source_refs_json={
            "curated_orientation": curated.source_refs,
            "profile_items": profile.source_refs,
        },
        diagnostics_json={},
        build_status="active",
        created_at="2026-06-08T09:00:00+00:00",
        updated_at="2026-06-08T09:00:00+00:00",
    )


def test_render_packages_places_curated_orientation_before_profile() -> None:
    result = _render_packages(
        [_record()],
        selected_source_keys=set(),
        live_topic_present=False,
        live_contract_present=False,
        live_state_present=False,
        include_recent_verbatim_seed=True,
        budget_tokens=500,
    )

    assert result.diagnostics["selected_curated_items"] == 1
    assert result.diagnostics["selected_profile_items"] == 1
    assert "[Curated Initial Orientation]" in result.block
    assert "[Prepared Memory Profile]" in result.block
    assert result.block.index("[Curated Initial Orientation]") < result.block.index(
        "[Prepared Memory Profile]"
    )
    assert "[historical] The small trust repair" in result.block


def test_render_packages_dedupes_curated_orientation_sources() -> None:
    result = _render_packages(
        [_record()],
        selected_source_keys={("memory", "mem_trust")},
        live_topic_present=False,
        live_contract_present=False,
        live_state_present=False,
        include_recent_verbatim_seed=True,
        budget_tokens=500,
    )

    assert result.diagnostics["selected_curated_items"] == 0
    assert result.diagnostics["dropped_curated_items"] == 1
    assert "[Curated Initial Orientation]" not in result.block
    assert "[Prepared Memory Profile]" in result.block


def test_render_packages_drops_recent_seed_only_curated_items_when_recent_seed_disabled() -> None:
    record = _record()
    message_curated = InitialContextPackageProfileItem(
        item_id="curated:recent",
        text="The user just asked about raw transcript details.",
        reason_category="recent_message_orientation",
        source_refs=[
            {
                "source_kind": "message",
                "message_id": "msg_1",
                "conversation_id": "cnv_1",
            }
        ],
        status="current",
        salience=0.8,
    )
    record.blocks_json.curated_items = [message_curated]
    record.source_refs_json["curated_orientation"] = message_curated.source_refs

    result = _render_packages(
        [record],
        selected_source_keys=set(),
        live_topic_present=False,
        live_contract_present=False,
        live_state_present=False,
        include_recent_verbatim_seed=False,
        budget_tokens=500,
    )

    assert result.diagnostics["selected_curated_items"] == 0
    assert result.diagnostics["dropped_curated_items"] == 1
    assert "raw transcript details" not in result.block
    assert "[Prepared Memory Profile]" in result.block


def test_render_packages_drops_mixed_recent_seed_curated_items_when_disabled() -> None:
    record = _record()
    mixed_curated = InitialContextPackageProfileItem(
        item_id="curated:mixed",
        text="A memory plus recent transcript detail should not render.",
        reason_category="mixed_recent_orientation",
        source_refs=[
            {"source_kind": "memory_object", "memory_id": "mem_profile"},
            {
                "source_kind": "message",
                "message_id": "msg_1",
                "conversation_id": "cnv_1",
            },
        ],
        status="current",
        salience=0.8,
    )
    record.blocks_json.curated_items = [mixed_curated]
    record.source_refs_json["curated_orientation"] = mixed_curated.source_refs

    result = _render_packages(
        [record],
        selected_source_keys=set(),
        live_topic_present=False,
        live_contract_present=False,
        live_state_present=False,
        include_recent_verbatim_seed=False,
        budget_tokens=500,
    )

    assert result.diagnostics["selected_curated_items"] == 0
    assert result.diagnostics["dropped_curated_items"] == 1
    assert "recent transcript detail" not in result.block
    assert "[Prepared Memory Profile]" in result.block
