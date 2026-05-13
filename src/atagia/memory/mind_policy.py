"""Mind perspective eligibility helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from atagia.models.schemas_memory import MindTopology, OverseerGrantKind, RetrievalPlan

OVERSEER_CONTEXT_GRANT_KINDS: tuple[str, ...] = (
    OverseerGrantKind.READ.value,
    OverseerGrantKind.SUMMARIZE.value,
    OverseerGrantKind.COORDINATE.value,
)


def mind_visibility_sql_clause(
    plan: RetrievalPlan,
    *,
    alias: str,
    owner_column: str = "memory_owner_id",
) -> tuple[str, list[Any]]:
    """Return SQL that enforces Mind visibility before candidate ranking."""

    return mind_visibility_sql_clause_for_context(
        active_mind_id=plan.active_mind_id,
        mind_topology=plan.mind_topology,
        alias=alias,
        owner_column=owner_column,
    )


def mind_visibility_sql_clause_for_context(
    *,
    active_mind_id: str | None,
    mind_topology: MindTopology | str | None,
    alias: str,
    owner_column: str = "memory_owner_id",
    user_column: str = "user_id",
    space_column: str = "space_id",
    realm_column: str = "realm_id",
    allow_overseer_grants: bool = True,
) -> tuple[str, list[Any]]:
    """Return SQL that enforces Mind visibility for a retrieval context."""

    prefix = f"{alias}." if alias else ""
    owner_expr = f"{prefix}{owner_column}"
    user_expr = f"{prefix}{user_column}"
    space_expr = f"{prefix}{space_column}"
    realm_expr = f"{prefix}{realm_column}"
    normalized_active_mind = _optional_text(active_mind_id)
    if normalized_active_mind is None:
        return f"({owner_expr} IS NULL)", []

    topology = _mind_topology(mind_topology)
    if topology is MindTopology.UNIMIND:
        return f"({owner_expr} IS NULL OR {owner_expr} = ?)", [normalized_active_mind]

    if topology is MindTopology.OJOCENTAURI:
        if not allow_overseer_grants:
            return f"({owner_expr} = ?)", [normalized_active_mind]
        grant_placeholders = ", ".join("?" for _ in OVERSEER_CONTEXT_GRANT_KINDS)
        as_of = datetime.now(timezone.utc).isoformat()
        return (
            "("
            f"{owner_expr} = ? "
            "OR EXISTS ("
            "SELECT 1 FROM overseer_grants AS og "
            f"WHERE og.owner_user_id = {user_expr} "
            "AND og.overseer_mind_id = ? "
            "AND EXISTS ("
            "SELECT 1 FROM minds AS og_overseer_mind "
            "WHERE og_overseer_mind.owner_user_id = og.owner_user_id "
            "AND og_overseer_mind.id = og.overseer_mind_id "
            "AND og_overseer_mind.kind = 'overseer'"
            ") "
            "AND og.revoked_at IS NULL "
            "AND (og.expires_at IS NULL OR og.expires_at > ?) "
            f"AND og.grant_kind IN ({grant_placeholders}) "
            "AND ("
            "("
            "og.target_kind = 'mind' "
            f"AND og.target_id = {owner_expr} "
            "AND EXISTS ("
            "SELECT 1 FROM minds AS og_target_mind "
            "WHERE og_target_mind.owner_user_id = og.owner_user_id "
            "AND og_target_mind.id = og.target_id"
            ")"
            ") "
            "OR ("
            "og.target_kind = 'space' "
            f"AND og.target_id = {space_expr} "
            "AND EXISTS ("
            "SELECT 1 FROM spaces AS og_target_space "
            "WHERE og_target_space.owner_user_id = og.owner_user_id "
            "AND og_target_space.id = og.target_id"
            ")"
            ") "
            "OR ("
            "og.target_kind = 'realm' "
            f"AND og.target_id = {realm_expr} "
            "AND EXISTS ("
            "SELECT 1 FROM realms AS og_target_realm "
            "WHERE og_target_realm.owner_user_id = og.owner_user_id "
            "AND og_target_realm.id = og.target_id"
            ")"
            ")"
            ")"
            ")"
            ")",
            [
                normalized_active_mind,
                normalized_active_mind,
                as_of,
                *OVERSEER_CONTEXT_GRANT_KINDS,
            ],
        )

    return f"({owner_expr} = ?)", [normalized_active_mind]


def candidate_allows_mind_boundary(
    candidate: dict[str, Any],
    plan: RetrievalPlan,
) -> bool:
    """Return whether a decoded candidate is visible to the active Mind."""

    candidate_owner_id = _optional_text(
        candidate.get("memory_owner_id") or candidate.get("active_mind_id")
    )
    active_mind_id = _optional_text(plan.active_mind_id)
    if active_mind_id is None:
        return candidate_owner_id is None

    topology = _mind_topology(plan.mind_topology)
    if topology is MindTopology.UNIMIND:
        return candidate_owner_id is None or candidate_owner_id == active_mind_id

    if topology is MindTopology.OJOCENTAURI:
        if candidate_owner_id == active_mind_id:
            return True
        return (
            _optional_text(candidate.get("mind_relation")) == "granted"
            and _optional_text(candidate.get("mind_grant_kind"))
            in OVERSEER_CONTEXT_GRANT_KINDS
        )

    return candidate_owner_id == active_mind_id


async def annotate_overseer_grants_for_rows(
    connection: Any,
    rows: list[dict[str, Any]],
    *,
    active_mind_id: str | None,
    mind_topology: MindTopology | str | None,
    owner_user_key: str = "user_id",
    owner_key: str = "memory_owner_id",
    space_key: str = "space_id",
    realm_key: str = "realm_id",
) -> None:
    """Annotate OjoCentauri rows with grants proven by overseer_grants."""

    if not rows or _mind_topology(mind_topology) is not MindTopology.OJOCENTAURI:
        return
    active_mind = _optional_text(active_mind_id)
    if active_mind is None:
        return

    owner_user_ids: set[str] = set()
    for row in rows:
        candidate_owner_id = _optional_text(row.get(owner_key))
        if candidate_owner_id == active_mind:
            row.setdefault("mind_relation", "same")
            row.setdefault("mind_grant_kind", "self")
            continue
        owner_user_id = _optional_text(row.get(owner_user_key))
        if owner_user_id is not None:
            owner_user_ids.add(owner_user_id)

    if not owner_user_ids:
        return

    grants_by_user: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    as_of = datetime.now(timezone.utc).isoformat()
    grant_placeholders = ", ".join("?" for _ in OVERSEER_CONTEXT_GRANT_KINDS)
    for owner_user_id in owner_user_ids:
        cursor = await connection.execute(
            f"""
            SELECT target_kind, target_id, grant_kind, visibility
            FROM overseer_grants
            WHERE owner_user_id = ?
              AND overseer_mind_id = ?
              AND EXISTS (
                  SELECT 1 FROM minds AS overseer_mind
                  WHERE overseer_mind.owner_user_id = overseer_grants.owner_user_id
                    AND overseer_mind.id = overseer_grants.overseer_mind_id
                    AND overseer_mind.kind = 'overseer'
              )
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
              AND grant_kind IN ({grant_placeholders})
              AND (
                  (
                      target_kind = 'mind'
                      AND EXISTS (
                          SELECT 1 FROM minds AS target_mind
                          WHERE target_mind.owner_user_id = overseer_grants.owner_user_id
                            AND target_mind.id = overseer_grants.target_id
                      )
                  )
                  OR (
                      target_kind = 'space'
                      AND EXISTS (
                          SELECT 1 FROM spaces AS target_space
                          WHERE target_space.owner_user_id = overseer_grants.owner_user_id
                            AND target_space.id = overseer_grants.target_id
                      )
                  )
                  OR (
                      target_kind = 'realm'
                      AND EXISTS (
                          SELECT 1 FROM realms AS target_realm
                          WHERE target_realm.owner_user_id = overseer_grants.owner_user_id
                            AND target_realm.id = overseer_grants.target_id
                      )
                  )
              )
            """,
            (
                owner_user_id,
                active_mind,
                as_of,
                *OVERSEER_CONTEXT_GRANT_KINDS,
            ),
        )
        grants_by_user[owner_user_id] = {
            (str(row["target_kind"]), str(row["target_id"])): dict(row)
            for row in await cursor.fetchall()
        }

    for row in rows:
        if _optional_text(row.get(owner_key)) == active_mind:
            continue
        owner_user_id = _optional_text(row.get(owner_user_key))
        if owner_user_id is None:
            continue
        grant = _matching_grant(
            row,
            grants_by_user.get(owner_user_id, {}),
            owner_key=owner_key,
            space_key=space_key,
            realm_key=realm_key,
        )
        if grant is None:
            continue
        row["mind_relation"] = "granted"
        row["mind_grant_kind"] = str(grant["grant_kind"])
        row["mind_grant_target_kind"] = str(grant["target_kind"])
        row["mind_grant_target_id"] = str(grant["target_id"])
        row["mind_grant_visibility"] = str(grant["visibility"])
        _annotate_mind_payload(
            row,
            active_mind_id=active_mind,
            owner_key=owner_key,
            grant=grant,
        )


def _mind_topology(value: Any) -> MindTopology:
    if isinstance(value, MindTopology):
        return value
    try:
        return MindTopology(str(value or MindTopology.UNIMIND.value))
    except ValueError:
        return MindTopology.UNIMIND


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _matching_grant(
    row: dict[str, Any],
    grants: dict[tuple[str, str], dict[str, Any]],
    *,
    owner_key: str,
    space_key: str,
    realm_key: str,
) -> dict[str, Any] | None:
    targets = (
        ("mind", _optional_text(row.get(owner_key))),
        ("space", _optional_text(row.get(space_key))),
        ("realm", _optional_text(row.get(realm_key))),
    )
    for target_kind, target_id in targets:
        if target_id is None:
            continue
        grant = grants.get((target_kind, target_id))
        if grant is not None:
            return grant
    return None


def _annotate_mind_payload(
    row: dict[str, Any],
    *,
    active_mind_id: str,
    owner_key: str,
    grant: dict[str, Any],
) -> None:
    payload_json = row.get("payload_json")
    if not isinstance(payload_json, dict):
        return
    mind_payload = payload_json.get("mind_perspective")
    if not isinstance(mind_payload, dict):
        mind_payload = {}
    owner_id = _optional_text(row.get(owner_key))
    if owner_id is not None:
        mind_payload.setdefault("memory_owner_id", owner_id)
    mind_payload["active_request_mind_id"] = active_mind_id
    mind_payload["mind_relation"] = "granted"
    mind_payload["grant_kind"] = str(grant["grant_kind"])
    mind_payload["grant_target_kind"] = str(grant["target_kind"])
    mind_payload["grant_target_id"] = str(grant["target_id"])
    payload_json["mind_perspective"] = mind_payload


__all__ = [
    "annotate_overseer_grants_for_rows",
    "candidate_allows_mind_boundary",
    "mind_visibility_sql_clause",
    "mind_visibility_sql_clause_for_context",
    "OVERSEER_CONTEXT_GRANT_KINDS",
]
