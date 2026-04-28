"""Operational profile loading, normalization, and trust policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atagia.core import json_utils
from atagia.core.canonical import canonical_json_hash
from atagia.models.schemas_memory import (
    OperationalCompute,
    OperationalConnectivity,
    OperationalIncidentScope,
    OperationalPower,
    OperationalProfile,
    OperationalProfileSnapshot,
    OperationalRiskLevel,
    OperationalSafety,
    OperationalSignals,
    ResolvedOperationalProfile,
)

EXPECTED_OPERATIONAL_PROFILE_IDS = frozenset(
    {"normal", "low_power", "offline", "emergency", "disaster"}
)
DEFAULT_OPERATIONAL_PROFILE_ID = "normal"


class UnknownOperationalProfileError(RuntimeError):
    """Raised when a requested operational profile is not configured."""


class OperationalProfileNotAuthorizedError(RuntimeError):
    """Raised when trust policy denies a requested operational profile."""


class OperationalTrustPolicy(BaseModel):
    """Minimal Phase 0 trust policy for operational profiles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_profiles: tuple[str, ...] = Field(
        default=("normal", "low_power", "offline"),
        min_length=1,
    )
    high_risk_enabled: bool = False
    trusted_local_mode: bool = True

    def authorize(
        self,
        *,
        profile_id: str,
        risk_level: OperationalRiskLevel,
    ) -> bool:
        if profile_id not in self.allowed_profiles:
            return False
        if risk_level is OperationalRiskLevel.HIGH_RISK and not self.high_risk_enabled:
            return False
        return True


class OperationalProfileLoader:
    """Filesystem-backed canonical operational profile loader."""

    def __init__(self, profiles_dir: Path) -> None:
        self._profiles_dir = profiles_dir
        self._profiles: dict[str, OperationalProfile] = {}

    def load_all(self) -> dict[str, OperationalProfile]:
        if not self._profiles_dir.exists():
            raise FileNotFoundError(f"Missing operational profiles directory: {self._profiles_dir}")

        profiles: dict[str, OperationalProfile] = {}
        for path in sorted(self._profiles_dir.glob("*.json")):
            payload = json_utils.loads(path.read_text(encoding="utf-8"))
            profile = OperationalProfile.model_validate(payload)
            profile_id = profile.operational_profile_id
            if profile_id != path.stem:
                raise ValueError(
                    f"Operational profile id {profile_id!r} must match filename {path.name!r}"
                )
            if profile_id in profiles:
                raise ValueError(f"Duplicate operational profile: {profile_id}")
            profile_hash = canonical_json_hash(profile.model_dump(mode="json"))
            profiles[profile_id] = profile.model_copy(update={"profile_hash": profile_hash})

        missing = EXPECTED_OPERATIONAL_PROFILE_IDS - profiles.keys()
        if missing:
            missing_profiles = ", ".join(sorted(missing))
            raise ValueError(f"Missing operational profiles: {missing_profiles}")
        extra = profiles.keys() - EXPECTED_OPERATIONAL_PROFILE_IDS
        if extra:
            extra_profiles = ", ".join(sorted(extra))
            raise ValueError(f"Unexpected operational profiles in Phase 0: {extra_profiles}")

        self._profiles = profiles
        return dict(profiles)

    def get(self, profile_id: str) -> OperationalProfile:
        if not self._profiles:
            self.load_all()
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise UnknownOperationalProfileError(
                f"Unknown operational profile: {profile_id}"
            ) from exc

    def resolve(
        self,
        *,
        operational_profile: str | None = None,
        operational_signals: OperationalSignals | dict[str, Any] | None = None,
        trust_policy: OperationalTrustPolicy | None = None,
    ) -> ResolvedOperationalProfile:
        """Normalize request profile/signals into an authorized snapshot."""

        profile_id = _normalize_profile_id(operational_profile)
        profile = self.get(profile_id)
        signal_model = _coerce_signals(operational_signals)
        effective_signals = _merge_signals(profile.signals, signal_model)
        effective_risk = _derive_risk(profile.risk_level, effective_signals)
        policy = trust_policy or OperationalTrustPolicy()
        authorized = policy.authorize(profile_id=profile_id, risk_level=effective_risk)
        if not authorized:
            raise OperationalProfileNotAuthorizedError(
                f"Operational profile is not authorized: {profile_id}"
            )
        profile_hash = profile.profile_hash or canonical_json_hash(profile.model_dump(mode="json"))
        token = _profile_token(
            profile_id=profile_id,
            signals=effective_signals,
            risk_level=effective_risk,
            authorized=authorized,
            profile_hash=profile_hash,
        )
        return ResolvedOperationalProfile(
            snapshot=OperationalProfileSnapshot(
                profile_id=profile_id,
                signals=effective_signals,
                risk_level=effective_risk,
                authorized=authorized,
                profile_hash=profile_hash,
                token=token,
            ),
            policy_override=profile.policy_override,
        )


def _normalize_profile_id(profile_id: str | None) -> str:
    normalized = (profile_id or DEFAULT_OPERATIONAL_PROFILE_ID).strip()
    return normalized or DEFAULT_OPERATIONAL_PROFILE_ID


def _coerce_signals(signals: OperationalSignals | dict[str, Any] | None) -> OperationalSignals:
    if signals is None:
        return OperationalSignals()
    if isinstance(signals, OperationalSignals):
        return signals
    return OperationalSignals.model_validate(signals)


def _merge_signals(base: OperationalSignals, explicit: OperationalSignals) -> OperationalSignals:
    payload = base.model_dump(mode="json")
    payload.update(explicit.model_dump(mode="json", exclude_none=True))
    return OperationalSignals.model_validate(payload)


def _derive_risk(
    profile_risk: OperationalRiskLevel,
    signals: OperationalSignals,
) -> OperationalRiskLevel:
    if (
        profile_risk is OperationalRiskLevel.HIGH_RISK
        or signals.safety in {OperationalSafety.HIGH_STAKES, OperationalSafety.EMERGENCY}
        or signals.incident_scope is OperationalIncidentScope.DISASTER
        or signals.power is OperationalPower.CRITICAL
    ):
        return OperationalRiskLevel.HIGH_RISK
    if (
        signals.connectivity is OperationalConnectivity.OFFLINE
        or signals.compute is OperationalCompute.LOCAL_ONLY
        or signals.incident_scope is OperationalIncidentScope.LOCAL_DISRUPTION
    ):
        return OperationalRiskLevel.SENSITIVE
    return profile_risk


def _profile_token(
    *,
    profile_id: str,
    signals: OperationalSignals,
    risk_level: OperationalRiskLevel,
    authorized: bool,
    profile_hash: str,
) -> str:
    return canonical_json_hash(
        {
            "profile_id": profile_id,
            "signals": signals.model_dump(mode="json"),
            "risk_level": risk_level.value,
            "authorized": authorized,
            "profile_hash": profile_hash,
        }
    )
