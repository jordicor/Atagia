"""Tests for operational profile loading and resolution."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from atagia.memory.operational_profile import (
    DEFAULT_OPERATIONAL_PROFILE_ID,
    EXPECTED_OPERATIONAL_PROFILE_IDS,
    OperationalProfileLoader,
    OperationalProfileNotAuthorizedError,
    OperationalTrustPolicy,
    UnknownOperationalProfileError,
)
from atagia.models.schemas_memory import (
    OperationalCompute,
    OperationalConnectivity,
    OperationalRiskLevel,
    OperationalSafety,
    OperationalSignals,
)

PROFILES_DIR = Path(__file__).resolve().parents[2] / "operational_profiles"


def _copy_profiles(tmp_path: Path) -> Path:
    target = tmp_path / "profiles"
    shutil.copytree(PROFILES_DIR, target)
    return target


def test_loads_canonical_phase_zero_profiles() -> None:
    profiles = OperationalProfileLoader(PROFILES_DIR).load_all()

    assert set(profiles) == set(EXPECTED_OPERATIONAL_PROFILE_IDS)
    assert profiles[DEFAULT_OPERATIONAL_PROFILE_ID].profile_hash
    assert profiles["offline"].signals.connectivity is OperationalConnectivity.OFFLINE


def test_loader_rejects_unknown_policy_fields(tmp_path: Path) -> None:
    profiles_dir = _copy_profiles(tmp_path)
    normal_path = profiles_dir / "normal.json"
    payload = json.loads(normal_path.read_text(encoding="utf-8"))
    payload["policy_override"]["privacy_ceiling"] = 1
    normal_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        OperationalProfileLoader(profiles_dir).load_all()


def test_loader_rejects_missing_extra_and_mismatched_presets(tmp_path: Path) -> None:
    missing_dir = _copy_profiles(tmp_path / "missing")
    (missing_dir / "offline.json").unlink()
    with pytest.raises(ValueError, match="Missing operational profiles"):
        OperationalProfileLoader(missing_dir).load_all()

    extra_dir = _copy_profiles(tmp_path / "extra")
    extra_payload = json.loads((extra_dir / "normal.json").read_text(encoding="utf-8"))
    extra_payload["operational_profile_id"] = "experimental"
    (extra_dir / "experimental.json").write_text(
        json.dumps(extra_payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unexpected operational profiles"):
        OperationalProfileLoader(extra_dir).load_all()

    mismatch_dir = _copy_profiles(tmp_path / "mismatch")
    mismatch_payload = json.loads((mismatch_dir / "normal.json").read_text(encoding="utf-8"))
    mismatch_payload["operational_profile_id"] = "offline"
    (mismatch_dir / "normal.json").write_text(
        json.dumps(mismatch_payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must match filename"):
        OperationalProfileLoader(mismatch_dir).load_all()


def test_resolves_normal_offline_and_signals_only_profiles() -> None:
    loader = OperationalProfileLoader(PROFILES_DIR)
    loader.load_all()

    normal = loader.resolve()
    explicit_normal = loader.resolve(operational_profile="normal")
    offline = loader.resolve(operational_profile="offline")
    signals_only = loader.resolve(
        operational_signals=OperationalSignals(compute=OperationalCompute.LOCAL_ONLY)
    )

    assert normal.snapshot.token == explicit_normal.snapshot.token
    assert normal.snapshot.profile_id == "normal"
    assert normal.snapshot.risk_level is OperationalRiskLevel.NORMAL
    assert offline.snapshot.risk_level is OperationalRiskLevel.SENSITIVE
    assert signals_only.snapshot.profile_id == "normal"
    assert signals_only.snapshot.risk_level is OperationalRiskLevel.SENSITIVE


def test_preset_and_explicit_signals_merge_field_by_field() -> None:
    loader = OperationalProfileLoader(PROFILES_DIR)
    loader.load_all()

    resolved = loader.resolve(
        operational_profile="offline",
        operational_signals={
            "power": "constrained",
            "compute": "local_only",
        },
    )

    assert resolved.snapshot.signals.connectivity is OperationalConnectivity.OFFLINE
    assert resolved.snapshot.signals.power.value == "constrained"
    assert resolved.snapshot.signals.compute is OperationalCompute.LOCAL_ONLY
    assert resolved.snapshot.risk_level is OperationalRiskLevel.SENSITIVE


def test_unknown_and_high_risk_profiles_fail_fast_by_policy() -> None:
    loader = OperationalProfileLoader(PROFILES_DIR)
    loader.load_all()

    with pytest.raises(UnknownOperationalProfileError):
        loader.resolve(operational_profile="space_station")

    with pytest.raises(OperationalProfileNotAuthorizedError):
        loader.resolve(operational_profile="emergency")

    with pytest.raises(OperationalProfileNotAuthorizedError):
        loader.resolve(
            operational_signals=OperationalSignals(safety=OperationalSafety.EMERGENCY)
        )

    emergency = loader.resolve(
        operational_profile="emergency",
        trust_policy=OperationalTrustPolicy(
            allowed_profiles=("normal", "low_power", "offline", "emergency"),
            high_risk_enabled=True,
        ),
    )
    assert emergency.snapshot.risk_level is OperationalRiskLevel.HIGH_RISK
    assert emergency.snapshot.signals.safety is OperationalSafety.EMERGENCY
