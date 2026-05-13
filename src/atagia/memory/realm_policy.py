"""Realm world/domain eligibility helpers."""

from __future__ import annotations

from typing import Any

from atagia.models.schemas_memory import CrossRealmMode, RetrievalPlan

ATTRIBUTED_REALM_BRIDGE_MODES: tuple[str, ...] = (
    CrossRealmMode.ATTRIBUTED.value,
    CrossRealmMode.APPLICABLE.value,
)
APPLICABLE_REALM_BRIDGE_MODES: tuple[str, ...] = (
    CrossRealmMode.APPLICABLE.value,
)


def realm_visibility_sql_clause(
    plan: RetrievalPlan,
    *,
    alias: str,
    realm_column: str = "realm_id",
    user_column: str = "user_id",
    allowed_bridge_modes: tuple[CrossRealmMode | str, ...] = ATTRIBUTED_REALM_BRIDGE_MODES,
) -> tuple[str, list[Any]]:
    """Return SQL that enforces Realm visibility before ranking."""

    return realm_visibility_sql_clause_for_context(
        active_realm_id=plan.active_realm_id,
        alias=alias,
        realm_column=realm_column,
        user_column=user_column,
        allowed_bridge_modes=allowed_bridge_modes,
    )


def realm_visibility_sql_clause_for_context(
    *,
    active_realm_id: str | None,
    alias: str,
    realm_column: str = "realm_id",
    user_column: str = "user_id",
    allowed_bridge_modes: tuple[CrossRealmMode | str, ...] = ATTRIBUTED_REALM_BRIDGE_MODES,
) -> tuple[str, list[Any]]:
    """Return SQL for same-Realm plus explicit bridge visibility."""

    prefix = f"{alias}." if alias else ""
    realm_expr = f"{prefix}{realm_column}"
    user_expr = f"{prefix}{user_column}"
    normalized_active_realm = _optional_text(active_realm_id)
    if normalized_active_realm is None:
        return f"({realm_expr} IS NULL)", []
    bridge_modes = _bridge_mode_values(allowed_bridge_modes)
    if not bridge_modes:
        return f"({realm_expr} IS NULL OR {realm_expr} = ?)", [normalized_active_realm]
    mode_placeholders = ", ".join("?" for _ in bridge_modes)
    return (
        "("
        f"{realm_expr} IS NULL "
        f"OR {realm_expr} = ? "
        "OR EXISTS ("
        "SELECT 1 FROM realm_bridges AS rb "
        f"WHERE rb.owner_user_id = {user_expr} "
        "AND rb.source_realm_id = ? "
        f"AND rb.target_realm_id = {realm_expr} "
        f"AND rb.cross_realm_mode IN ({mode_placeholders})"
        ")"
        ")",
        [normalized_active_realm, normalized_active_realm, *bridge_modes],
    )


def candidate_allows_realm_boundary(
    candidate: dict[str, Any],
    plan: RetrievalPlan,
) -> bool:
    """Return whether a decoded candidate is visible after bridge annotation."""

    candidate_realm_id = _optional_text(
        candidate.get("realm_id") or candidate.get("active_realm_id")
    )
    active_realm_id = _optional_text(plan.active_realm_id)
    if active_realm_id is None:
        return candidate_realm_id is None
    if candidate_realm_id is None or candidate_realm_id == active_realm_id:
        return True
    bridge_relation = _optional_text(candidate.get("realm_relation"))
    bridge_mode = _optional_text(candidate.get("realm_bridge_mode"))
    return (
        bridge_relation == "cross"
        and bridge_mode in ATTRIBUTED_REALM_BRIDGE_MODES
    )


async def annotate_realm_bridge_modes_for_rows(
    connection: Any,
    rows: list[dict[str, Any]],
    *,
    active_realm_id: str | None,
    realm_keys: tuple[str, ...] = ("realm_id", "active_realm_id"),
    owner_user_key: str = "user_id",
) -> None:
    """Annotate rows with bridge modes proven by explicit realm_bridges rows."""

    active_realm = _optional_text(active_realm_id)
    if not rows:
        return

    bridge_targets: set[tuple[str, str]] = set()
    for row in rows:
        realm_id = _first_text(row, realm_keys)
        if realm_id is None:
            row.setdefault("realm_relation", "unscoped")
            continue
        if active_realm is not None and realm_id == active_realm:
            row.setdefault("realm_relation", "same")
            row.setdefault("realm_bridge_mode", "same")
            continue
        if active_realm is None:
            continue
        owner_user_id = _optional_text(row.get(owner_user_key))
        if owner_user_id is not None:
            bridge_targets.add((owner_user_id, realm_id))

    if not bridge_targets or active_realm is None:
        return

    mode_by_target: dict[tuple[str, str], str] = {}
    grouped_targets: dict[str, list[str]] = {}
    for owner_user_id, target_realm_id in bridge_targets:
        grouped_targets.setdefault(owner_user_id, []).append(target_realm_id)
    for owner_user_id, target_realm_ids in grouped_targets.items():
        placeholders = ", ".join("?" for _ in target_realm_ids)
        cursor = await connection.execute(
            f"""
            SELECT target_realm_id, cross_realm_mode
            FROM realm_bridges
            WHERE owner_user_id = ?
              AND source_realm_id = ?
              AND target_realm_id IN ({placeholders})
              AND cross_realm_mode IN ('attributed', 'applicable')
            """,
            (owner_user_id, active_realm, *target_realm_ids),
        )
        for bridge_row in await cursor.fetchall():
            mode_by_target[
                (owner_user_id, str(bridge_row["target_realm_id"]))
            ] = str(bridge_row["cross_realm_mode"])

    for row in rows:
        realm_id = _first_text(row, realm_keys)
        owner_user_id = _optional_text(row.get(owner_user_key))
        if realm_id is None or owner_user_id is None:
            continue
        bridge_mode = mode_by_target.get((owner_user_id, realm_id))
        if bridge_mode is None:
            continue
        row["realm_relation"] = "cross"
        row["realm_bridge_mode"] = bridge_mode
        payload_json = row.get("payload_json")
        if isinstance(payload_json, dict):
            realm_payload = payload_json.get("realm")
            if not isinstance(realm_payload, dict):
                realm_payload = {}
            realm_payload.setdefault("active_realm_id", realm_id)
            realm_payload["cross_realm_mode"] = bridge_mode
            payload_json["realm"] = realm_payload


def _bridge_mode_values(
    values: tuple[CrossRealmMode | str, ...],
) -> list[str]:
    normalized: list[str] = []
    for value in values:
        try:
            mode = CrossRealmMode(value)
        except ValueError:
            continue
        if mode is CrossRealmMode.NONE:
            continue
        if mode.value not in normalized:
            normalized.append(mode.value)
    return normalized


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _optional_text(row.get(key))
        if value is not None:
            return value
    return None


__all__ = [
    "annotate_realm_bridge_modes_for_rows",
    "candidate_allows_realm_boundary",
    "APPLICABLE_REALM_BRIDGE_MODES",
    "ATTRIBUTED_REALM_BRIDGE_MODES",
    "realm_visibility_sql_clause",
    "realm_visibility_sql_clause_for_context",
]
