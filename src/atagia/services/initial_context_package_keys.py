"""Shared key construction for prepared initial-context packages."""

from __future__ import annotations

from typing import Any

from atagia.models.schemas_initial_context_package import (
    InitialContextPackageCoordinateSignature,
    InitialContextPackageKey,
    InitialContextPackageKind,
    InitialContextPackagePolicySignature,
)
from atagia.models.schemas_memory import OperationalProfileSnapshot


def initial_context_package_subject(
    *,
    user_persona_id: str | None,
    platform_id: str | None,
    character_id: str | None,
    workspace_id: str | None,
    assistant_mode_id: str,
    mode: str | None,
) -> dict[str, str | None]:
    """Return the stable subject fields stored in package keys."""

    return {
        "user_persona_id": user_persona_id,
        "platform_id": platform_id,
        "character_id": character_id,
        "workspace_id": workspace_id,
        "assistant_mode_id": assistant_mode_id,
        "mode": mode or assistant_mode_id,
    }


def build_initial_context_package_key(
    *,
    version: int,
    package_kind: InitialContextPackageKind,
    user_id: str,
    conversation_id: str | None,
    retrieval_profile_id: str,
    subject_json: dict[str, Any],
    policy_signature: InitialContextPackagePolicySignature,
    coordinate_signature: InitialContextPackageCoordinateSignature,
    operational_profile: OperationalProfileSnapshot | None,
) -> InitialContextPackageKey:
    """Return the canonical key used by both refresh and prompt reads."""

    return InitialContextPackageKey(
        version=version,
        package_kind=package_kind,
        user_id=user_id,
        conversation_id=conversation_id,
        retrieval_profile_id=retrieval_profile_id,
        subject_json=dict(subject_json),
        policy_json={
            "effective_policy_hash": policy_signature.effective_policy_hash,
            "policy_prompt_hash": policy_signature.policy_prompt_hash,
            "privacy_enforcement": policy_signature.privacy_enforcement,
            "authority": policy_signature.authority_json,
        },
        coordinate_json={
            "coordinate_signature_hash": (
                coordinate_signature.coordinate_signature_hash
            ),
            "complete": coordinate_signature.complete,
            "markers": _key_coordinate_markers(coordinate_signature.markers_json),
        },
        operational_json={
            "operational_profile": (
                operational_profile.model_dump(mode="json")
                if operational_profile is not None
                else None
            )
        },
    )


def _key_coordinate_markers(markers: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity": markers.get("identity") or {},
        "lifecycle": markers.get("lifecycle") or {},
        "presence": _row_identity(markers.get("presence")),
        "space": _row_identity(markers.get("space")),
        "mind": {
            "topology": (markers.get("mind") or {}).get("topology"),
            "row": _row_identity((markers.get("mind") or {}).get("row")),
        },
        "ojocentauri": markers.get("ojocentauri") or {},
        "embodiment": _row_identity(markers.get("embodiment")),
        "realm": {
            "row": _row_identity((markers.get("realm") or {}).get("row")),
            "bridges": (markers.get("realm") or {}).get("bridges") or [],
        },
        "missing": list(markers.get("missing") or []),
        "complete": bool(markers.get("complete")),
    }


def _row_identity(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    return {
        key: row.get(key)
        for key in (
            "id",
            "owner_user_id",
            "kind",
            "display_name",
            "boundary_mode",
            "cross_embodiment_mode",
            "cross_realm_mode",
            "updated_at",
        )
        if key in row
    }
