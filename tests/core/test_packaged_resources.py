"""Inventory and parity guards for canonical packaged resources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atagia.core.config import Settings, default_resource_path
from atagia.core.db_sqlite import MigrationManager
from atagia.memory.operational_profile import (
    EXPECTED_OPERATIONAL_PROFILE_IDS,
    OperationalProfileLoader,
)
from atagia.memory.policy_manifest import (
    EXPECTED_RETRIEVAL_PROFILE_IDS,
    ManifestLoader,
    PolicyResolver,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolved_policies(manifests_dir: Path) -> dict[str, dict[str, object]]:
    manifests = ManifestLoader(manifests_dir).load_all()
    resolver = PolicyResolver()
    return {
        mode_id: resolver.resolve(manifest, None, None).model_dump(mode="json")
        for mode_id, manifest in sorted(manifests.items())
    }


def _resolved_operational_profiles(
    profiles_dir: Path,
) -> dict[str, dict[str, object]]:
    return {
        profile_id: profile.model_dump(mode="json")
        for profile_id, profile in sorted(
            OperationalProfileLoader(profiles_dir).load_all().items()
        )
    }


def test_canonical_resource_inventory_is_complete_and_contiguous() -> None:
    migrations_dir = Path(default_resource_path("migrations"))
    migrations = MigrationManager(migrations_dir).discover()
    versions = [migration.version for migration in migrations]

    assert versions == list(range(1, max(versions) + 1))
    assert {53, 54}.issubset(versions)
    assert {migration.path.name for migration in migrations} == {
        path.name for path in migrations_dir.glob("*.sql")
    }

    manifests = ManifestLoader(Path(default_resource_path("manifests"))).load_all()
    profiles = OperationalProfileLoader(
        Path(default_resource_path("operational_profiles"))
    ).load_all()
    assert set(manifests) == EXPECTED_RETRIEVAL_PROFILE_IDS
    assert set(profiles) == EXPECTED_OPERATIONAL_PROFILE_IDS


def test_approved_manifest_context_item_limits_are_shipped() -> None:
    manifests = ManifestLoader(Path(default_resource_path("manifests"))).load_all()
    for mode_id, manifest in manifests.items():
        expected = 7 if mode_id == "intimacy" else 12
        assert manifest.retrieval_params.final_context_items == expected


def test_default_resources_are_cwd_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    before_policies = _resolved_policies(Path(default_resource_path("manifests")))
    before_profiles = _resolved_operational_profiles(
        Path(default_resource_path("operational_profiles"))
    )
    monkeypatch.chdir(tmp_path)
    for variable in (
        "ATAGIA_MIGRATIONS_PATH",
        "ATAGIA_MANIFESTS_PATH",
        "ATAGIA_OPERATIONAL_PROFILES_PATH",
    ):
        monkeypatch.delenv(variable, raising=False)
    settings = Settings.from_env()

    assert _resolved_policies(settings.manifests_dir()) == before_policies
    assert (
        _resolved_operational_profiles(settings.operational_profiles_dir())
        == before_profiles
    )
    assert MigrationManager(settings.migrations_dir()).discover()


def test_no_duplicate_root_production_resources_exist() -> None:
    duplicates = {
        name: REPO_ROOT / name
        for name in ("migrations", "manifests", "operational_profiles")
        if (REPO_ROOT / name).exists()
    }
    assert not duplicates, json.dumps(
        {name: str(path) for name, path in duplicates.items()},
        sort_keys=True,
    )
