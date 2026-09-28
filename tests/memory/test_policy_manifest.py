"""Tests for retrieval profile manifests and policy resolution."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.memory.policy_manifest import (
    DEFAULT_RETRIEVAL_SCOPE_FILTER,
    ManifestLoader,
    PolicyResolver,
    ResolvedRetrievalPolicy,
    resolved_policy_provenance,
    sync_assistant_modes,
)
from atagia.models.schemas_memory import MemoryScope, OperationalPolicyOverride

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"


def _copy_manifests_to(tmp_path: Path) -> Path:
    target = tmp_path / "manifests"
    target.mkdir()
    for source in MANIFESTS_DIR.glob("*.json"):
        target.joinpath(source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _load_manifest_json(directory: Path, filename: str) -> dict[str, object]:
    return json.loads(directory.joinpath(filename).read_text(encoding="utf-8"))


def _write_manifest_json(directory: Path, filename: str, payload: dict[str, object]) -> None:
    directory.joinpath(filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_load_all_manifests_successfully() -> None:
    loader = ManifestLoader(MANIFESTS_DIR)

    manifests = loader.load_all()

    assert set(manifests) == {
        "biographical_interview",
        "brainstorm",
        "coding_debug",
        "companion",
        "general_qa",
        "intimacy",
        "personal_assistant",
        "research_deep_dive",
    }
    assert manifests["coding_debug"].privacy_ceiling == 1
    assert manifests["coding_debug"].profile_id.value == "coding_debug"
    assert manifests["coding_debug"].transcript_budget_tokens == 8000
    assert manifests["coding_debug"].context_cache_policy.max_messages_without_refresh == 5
    assert manifests["general_qa"].allow_intimacy_context is False
    assert manifests["intimacy"].allow_intimacy_context is True
    assert manifests["general_qa"].context_cache_policy.base_ttl_seconds == 900
    assert manifests["personal_assistant"].privacy_ceiling == 3
    assert manifests["research_deep_dive"].prompt_hash is not None


def test_loader_rejects_invalid_manifest_values(tmp_path: Path) -> None:
    manifests_dir = _copy_manifests_to(tmp_path)
    payload = _load_manifest_json(manifests_dir, "coding_debug.json")
    payload["context_cache_policy"]["sync_threshold"] = 1.5
    _write_manifest_json(manifests_dir, "coding_debug.json", payload)

    loader = ManifestLoader(manifests_dir)

    with pytest.raises(ValueError):
        loader.load_all()


def test_prompt_hash_changes_when_manifest_content_changes(tmp_path: Path) -> None:
    manifests_dir = _copy_manifests_to(tmp_path)
    loader = ManifestLoader(manifests_dir)
    original_hash = loader.load_all()["coding_debug"].prompt_hash

    payload = _load_manifest_json(manifests_dir, "coding_debug.json")
    payload["context_budget_tokens"] = 5400
    _write_manifest_json(manifests_dir, "coding_debug.json", payload)

    updated_hash = ManifestLoader(manifests_dir).load_all()["coding_debug"].prompt_hash

    assert updated_hash is not None
    assert original_hash != updated_hash


def test_policy_resolution_without_overrides_returns_manifest_values() -> None:
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]

    resolved = PolicyResolver().resolve(manifest, None, None)

    assert resolved.profile_id == manifest.profile_id
    assert resolved.allowed_scopes == DEFAULT_RETRIEVAL_SCOPE_FILTER
    assert MemoryScope.ASSISTANT_MODE not in resolved.allowed_scopes
    assert resolved.privacy_ceiling == manifest.privacy_ceiling
    assert resolved.context_budget_tokens == manifest.context_budget_tokens
    assert resolved.transcript_budget_tokens == manifest.transcript_budget_tokens
    assert resolved.retrieval_params == manifest.retrieval_params
    assert resolved.context_cache_policy == manifest.context_cache_policy


def test_policy_resolution_workspace_override_restricts_privacy_ceiling() -> None:
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["companion"]

    resolved = PolicyResolver().resolve(
        manifest,
        {"privacy_ceiling": 1},
        None,
    )

    assert resolved.privacy_ceiling == 1
    assert resolved.cross_chat_allowed is True
    assert resolved.allow_intimacy_context is False


def test_policy_resolution_conversation_override_takes_preference_values() -> None:
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["research_deep_dive"]

    resolved = PolicyResolver().resolve(
        manifest,
        {
            "context_budget_tokens": 5000,
            "transcript_budget_tokens": 7000,
            "retrieval_params": {"fts_limit": 18, "rerank_top_k": 12},
        },
        {
            "context_budget_tokens": 3200,
            "transcript_budget_tokens": 2800,
            "retrieval_params": {"fts_limit": 9},
            "preferred_memory_types": ["evidence", "summary_view"],
        },
    )

    assert resolved.context_budget_tokens == 3200
    assert resolved.transcript_budget_tokens == 2800
    assert resolved.retrieval_params.fts_limit == 9
    assert resolved.retrieval_params.rerank_top_k == 12
    assert [item.value for item in resolved.preferred_memory_types] == ["evidence", "summary_view"]


def test_allowed_scopes_are_not_retrieval_profile_gates() -> None:
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]

    resolved = PolicyResolver().resolve(
        manifest,
        None,
        None,
        OperationalPolicyOverride(allowed_scopes=[MemoryScope.CONVERSATION]),
    )

    assert resolved.allowed_scopes == DEFAULT_RETRIEVAL_SCOPE_FILTER
    assert MemoryScope.ASSISTANT_MODE not in resolved.allowed_scopes


def test_prompt_hash_changes_when_context_cache_policy_changes(tmp_path: Path) -> None:
    manifests_dir = _copy_manifests_to(tmp_path)
    loader = ManifestLoader(manifests_dir)
    original_hash = loader.load_all()["general_qa"].prompt_hash

    payload = _load_manifest_json(manifests_dir, "general_qa.json")
    payload["context_cache_policy"]["base_ttl_seconds"] = 1200
    _write_manifest_json(manifests_dir, "general_qa.json", payload)

    updated_hash = ManifestLoader(manifests_dir).load_all()["general_qa"].prompt_hash

    assert updated_hash is not None
    assert original_hash != updated_hash


@pytest.mark.asyncio
async def test_sync_assistant_modes_upserts_manifest_rows() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 17, 0, tzinfo=timezone.utc))
    try:
        manifests = ManifestLoader(MANIFESTS_DIR).load_all()

        await sync_assistant_modes(connection, manifests, clock)

        cursor = await connection.execute(
            """
            SELECT id, display_name, prompt_hash, memory_policy_json, created_at, updated_at
            FROM assistant_modes
            ORDER BY id ASC
            """
        )
        rows = await cursor.fetchall()

        assert len(rows) == 8
        coding_debug_row = next(row for row in rows if row["id"] == "coding_debug")
        assert coding_debug_row["prompt_hash"] == manifests["coding_debug"].prompt_hash
        assert json.loads(coding_debug_row["memory_policy_json"]) == manifests["coding_debug"].model_dump(mode="json")
        assert coding_debug_row["created_at"] == "2026-03-30T17:00:00+00:00"
        assert coding_debug_row["updated_at"] == "2026-03-30T17:00:00+00:00"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_sync_assistant_modes_updates_existing_rows(tmp_path: Path) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 17, 0, tzinfo=timezone.utc))
    try:
        manifests_dir = _copy_manifests_to(tmp_path)
        original_manifests = ManifestLoader(manifests_dir).load_all()
        await sync_assistant_modes(connection, original_manifests, clock)

        payload = _load_manifest_json(manifests_dir, "brainstorm.json")
        payload["display_name"] = "Brainstorming"
        _write_manifest_json(manifests_dir, "brainstorm.json", payload)

        clock.advance(seconds=90)
        updated_manifests = ManifestLoader(manifests_dir).load_all()
        await sync_assistant_modes(connection, updated_manifests, clock)

        cursor = await connection.execute(
            """
            SELECT display_name, prompt_hash, created_at, updated_at
            FROM assistant_modes
            WHERE id = 'brainstorm'
            """
        )
        row = await cursor.fetchone()

        assert row["display_name"] == "Brainstorming"
        assert row["prompt_hash"] == updated_manifests["brainstorm"].prompt_hash
        assert row["prompt_hash"] != original_manifests["brainstorm"].prompt_hash
        assert row["created_at"] == "2026-03-30T17:00:00+00:00"
        assert row["updated_at"] == "2026-03-30T17:01:30+00:00"
    finally:
        await connection.close()


def test_resolved_policy_provenance_names_the_layer_that_produced_each_field(
    tmp_path: Path,
) -> None:
    """Provenance is a claim about the manifest FILE, not about the schema.

    Every resolved field used to be tagged ``manifest``, including fields no
    manifest declares and fields no manifest even CAN declare. That attributes a
    value to a source that did not produce it, which is the exact defect class
    the effective-settings report exists to remove.
    """
    manifests_dir = _copy_manifests_to(tmp_path)
    payload = _load_manifest_json(manifests_dir, "general_qa.json")
    assert "allow_intimacy_context" not in payload
    manifest = ManifestLoader(manifests_dir).get("general_qa")

    provenance = resolved_policy_provenance(manifest)

    assert set(provenance) == set(ResolvedRetrievalPolicy.model_fields)
    # Declared in the file.
    for declared in (
        "profile_id",
        "display_name",
        "privacy_ceiling",
        "context_budget_tokens",
        "transcript_budget_tokens",
        "retrieval_params",
        "context_cache_policy",
    ):
        assert provenance[declared] == "manifest", declared
    # Schema default: the manifest owns the field but this file leaves it unset.
    assert provenance["allow_intimacy_context"] == "default"
    # The resolver decides these outright; no manifest field feeds them.
    for resolver_owned in (
        "cross_chat_allowed",
        "allowed_scopes",
        "allow_private_sensitivity",
    ):
        assert provenance[resolver_owned] == "resolver", resolver_owned
    # Derived from the manifest payload rather than declared in it.
    assert provenance["prompt_hash"] == "computed"


def test_declaring_an_optional_field_flips_its_provenance_to_manifest(
    tmp_path: Path,
) -> None:
    """The tag follows the file: declaring the field the previous test found
    unset makes it manifest-sourced, even when the declared value equals the
    schema default (configuring something is not the same as leaving it out)."""
    manifests_dir = _copy_manifests_to(tmp_path)
    payload = _load_manifest_json(manifests_dir, "general_qa.json")
    payload["allow_intimacy_context"] = False
    _write_manifest_json(manifests_dir, "general_qa.json", payload)

    manifest = ManifestLoader(manifests_dir).get("general_qa")

    assert manifest.allow_intimacy_context is False
    assert resolved_policy_provenance(manifest)["allow_intimacy_context"] == "manifest"
