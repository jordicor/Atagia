"""Signature, freshness, and cache invalidation helpers for context packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiosqlite
from pydantic import BaseModel

from atagia.core import json_utils
from atagia.core.canonical import canonical_json_hash
from atagia.core.clock import Clock
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.core.mind_repository import DEFAULT_OVERSEER_MIND_ID
from atagia.core.storage_backend import StorageBackend
from atagia.memory.lifecycle_runner import cache_generation_key
from atagia.memory.policy_manifest import (
    ResolvedRetrievalPolicy,
    compute_effective_policy_hash,
)
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageCoordinateSignature,
    InitialContextPackageKind,
    InitialContextPackagePolicySignature,
    InitialContextPackageSourceFingerprint,
)
from atagia.models.schemas_memory import OperationalProfileSnapshot
from atagia.services.prompt_authority import PromptAuthorityContext


@dataclass(frozen=True, slots=True)
class InitialContextPackageCacheInvalidationResult:
    """Result of invalidating package rows and dependent context-cache entries."""

    stale_package_count: int
    deleted_context_views: int
    deleted_recent_windows: int
    cache_generation: int


def build_initial_context_package_policy_signature(
    resolved_policy: ResolvedRetrievalPolicy,
    *,
    privacy_enforcement: str = "enforce",
    authority_context: PromptAuthorityContext | None = None,
    operational_profile: OperationalProfileSnapshot | None = None,
) -> InitialContextPackagePolicySignature:
    """Return a stable signature for policy and authenticated authority inputs."""

    effective_privacy = (
        authority_context.effective_privacy_enforcement
        if authority_context is not None
        else privacy_enforcement
    )
    authority_json = (
        {
            "privacy_enforcement": authority_context.privacy_enforcement,
            "effective_privacy_enforcement": authority_context.effective_privacy_enforcement,
            "authenticated_privilege_level": authority_context.normalized_privilege_level,
            "authenticated_atagia_master": authority_context.authenticated_user_is_atagia_master,
            "trusted_evaluation": authority_context.trusted_evaluation,
            "authority_source": authority_context.authority_source,
            "purpose": authority_context.purpose,
        }
        if authority_context is not None
        else {
            "privacy_enforcement": privacy_enforcement,
            "effective_privacy_enforcement": privacy_enforcement,
            "authenticated_privilege_level": "standard",
            "authenticated_atagia_master": False,
            "trusted_evaluation": False,
            "authority_source": "unspecified",
            "purpose": None,
        }
    )
    policy_payload = resolved_policy.model_dump(mode="json")
    markers = {
        "version": 1,
        "effective_policy_hash": compute_effective_policy_hash(resolved_policy),
        "policy_prompt_hash": resolved_policy.prompt_hash,
        "privacy_enforcement": effective_privacy,
        "retrieval_profile_id": resolved_policy.profile_id.value,
        "policy": policy_payload,
        "authority": authority_json,
        "operational_profile": (
            operational_profile.model_dump(mode="json")
            if operational_profile is not None
            else None
        ),
    }
    return InitialContextPackagePolicySignature(
        effective_policy_hash=str(markers["effective_policy_hash"]),
        policy_prompt_hash=resolved_policy.prompt_hash,
        privacy_enforcement=effective_privacy,
        authority_json=authority_json,
        markers_json=markers,
    )


async def build_initial_context_package_coordinate_signature(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    retrieval_profile_id: str,
    conversation_id: str | None = None,
    conversation: Mapping[str, Any] | None = None,
    user_persona_id: str | None = None,
    platform_id: str | None = None,
    character_id: str | None = None,
    workspace_id: str | None = None,
    assistant_mode_id: str | None = None,
    active_presence_id: str | None = None,
    active_space_id: str | None = None,
    active_mind_id: str | None = None,
    mind_topology: str | None = None,
    active_embodiment_id: str | None = None,
    active_realm_id: str | None = None,
    incognito: bool | None = None,
    isolated_mode: bool | None = None,
    temporary: bool | None = None,
    purge_on_close: bool | None = None,
    now: datetime | None = None,
) -> InitialContextPackageCoordinateSignature:
    """Build deterministic markers for all current coordinate boundaries."""

    now_utc = _coerce_aware_utc(now or datetime.now(tz=timezone.utc))
    conversation_row = await _resolve_conversation(
        connection,
        user_id=user_id,
        conversation_id=conversation_id,
        conversation=conversation,
    )
    conversation_marker: dict[str, Any] | None = None
    if conversation_row is not None:
        conversation_marker = _conversation_marker(conversation_row)
        user_persona_id = _coalesce(
            user_persona_id,
            conversation_row.get("user_persona_id"),
        )
        platform_id = _coalesce(platform_id, conversation_row.get("platform_id"))
        character_id = _coalesce(character_id, conversation_row.get("character_id"))
        workspace_id = _coalesce(workspace_id, conversation_row.get("workspace_id"))
        assistant_mode_id = _coalesce(
            assistant_mode_id,
            conversation_row.get("assistant_mode_id"),
        )
        active_presence_id = _coalesce(
            active_presence_id,
            conversation_row.get("active_presence_id"),
        )
        active_space_id = _coalesce(active_space_id, conversation_row.get("active_space_id"))
        active_mind_id = _coalesce(active_mind_id, conversation_row.get("active_mind_id"))
        mind_topology = _coalesce(mind_topology, conversation_row.get("mind_topology"))
        active_embodiment_id = _coalesce(
            active_embodiment_id,
            conversation_row.get("active_embodiment_id"),
        )
        active_realm_id = _coalesce(
            active_realm_id,
            conversation_row.get("active_realm_id"),
        )
        incognito = _coalesce_bool(incognito, conversation_row.get("incognito"))
        isolated_mode = _coalesce_bool(isolated_mode, conversation_row.get("isolated_mode"))
        temporary = _coalesce_bool(temporary, conversation_row.get("temporary"))
        purge_on_close = _coalesce_bool(
            purge_on_close,
            conversation_row.get("purge_on_close"),
        )

    missing: list[str] = []
    if conversation_id is not None and conversation_row is None:
        missing.append("conversation")
    user_preferences = await _fetch_user_preferences(connection, user_id=user_id)
    if user_preferences is None:
        missing.append("user")

    presence = await _coordinate_row(
        connection,
        table="presences",
        user_id=user_id,
        id_value=active_presence_id,
        fields=(
            "id",
            "owner_user_id",
            "kind",
            "display_name",
            "source_kind",
            "source_id",
            "presence_cluster_id",
            "metadata_json",
            "updated_at",
        ),
    )
    if active_presence_id is not None and presence is None:
        missing.append("presence")

    space = await _coordinate_row(
        connection,
        table="spaces",
        user_id=user_id,
        id_value=active_space_id,
        fields=(
            "id",
            "owner_user_id",
            "boundary_mode",
            "display_name",
            "source_kind",
            "source_id",
            "metadata_json",
            "updated_at",
        ),
    )
    if active_space_id is not None and space is None:
        missing.append("space")

    mind = await _coordinate_row(
        connection,
        table="minds",
        user_id=user_id,
        id_value=active_mind_id,
        fields=(
            "id",
            "owner_user_id",
            "kind",
            "display_name",
            "source_kind",
            "source_id",
            "metadata_json",
            "updated_at",
        ),
    )
    if active_mind_id is not None and mind is None:
        missing.append("mind")

    embodiment = await _coordinate_row(
        connection,
        table="embodiments",
        user_id=user_id,
        id_value=active_embodiment_id,
        fields=(
            "id",
            "owner_user_id",
            "display_name",
            "source_kind",
            "source_id",
            "cross_embodiment_mode",
            "metadata_json",
            "updated_at",
        ),
    )
    if active_embodiment_id is not None and embodiment is None:
        missing.append("embodiment")

    realm = await _coordinate_row(
        connection,
        table="realms",
        user_id=user_id,
        id_value=active_realm_id,
        fields=(
            "id",
            "owner_user_id",
            "display_name",
            "source_kind",
            "source_id",
            "cross_realm_mode",
            "metadata_json",
            "updated_at",
        ),
    )
    if active_realm_id is not None and realm is None:
        missing.append("realm")

    grants = await _fetch_overseer_grants(
        connection,
        user_id=user_id,
        active_mind_id=active_mind_id,
        mind_topology=mind_topology,
        now=now_utc,
    )
    realm_bridges = await _fetch_realm_bridges(
        connection,
        user_id=user_id,
        active_realm_id=active_realm_id,
    )
    markers = {
        "version": 1,
        "user_id": user_id,
        "retrieval_profile_id": retrieval_profile_id,
        "conversation_id": (
            str(conversation_row["id"])
            if conversation_row is not None
            else conversation_id
        ),
        "identity": {
            "assistant_mode_id": assistant_mode_id,
            "workspace_id": workspace_id,
            "user_persona_id": user_persona_id,
            "platform_id": platform_id,
            "character_id": character_id,
        },
        "lifecycle": {
            "incognito": bool(incognito) if incognito is not None else False,
            "isolated_mode": bool(isolated_mode) if isolated_mode is not None else False,
            "temporary": bool(temporary) if temporary is not None else False,
            "purge_on_close": (
                bool(purge_on_close) if purge_on_close is not None else False
            ),
        },
        "conversation": conversation_marker,
        "memory_preferences": user_preferences,
        "presence": presence,
        "space": space,
        "mind": {
            "topology": mind_topology or "unimind",
            "row": mind,
        },
        "ojocentauri": {
            "overseer_mind_id": _active_overseer_mind_id(
                active_mind_id=active_mind_id,
                mind_topology=mind_topology,
            ),
            "grants": grants,
            "next_expiry_at": _next_grant_expiry(grants),
        },
        "embodiment": embodiment,
        "realm": {
            "row": realm,
            "bridges": realm_bridges,
        },
        "missing": sorted(missing),
        "complete": not missing,
    }
    signature_hash = canonical_json_hash(markers)
    return InitialContextPackageCoordinateSignature(
        coordinate_signature_hash=signature_hash,
        complete=not missing,
        markers_json=markers,
    )


async def build_initial_context_package_source_fingerprint(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    conversation_id: str | None = None,
) -> InitialContextPackageSourceFingerprint:
    """Return aggregate source freshness markers for package materialization."""

    sources: dict[str, Any] = {
        "user": await _single_row_marker(
            connection,
            """
            SELECT updated_at,
                   deleted_at,
                   remember_across_chats,
                   remember_across_devices,
                   memory_privacy_mode
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ),
        "memory_objects": await _aggregate_memory_objects(
            connection,
            user_id=user_id,
        ),
        "belief_versions": await _aggregate_joined_to_memory(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(bv.created_at) AS max_created_at,
                   MAX(bv.version) AS max_version
            FROM belief_versions AS bv
            JOIN memory_objects AS mo ON mo.id = bv.belief_id
            WHERE mo.user_id = ?
            """,
            (user_id,),
        ),
        "memory_links": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(created_at) AS max_created_at
            FROM memory_links
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "memory_retrieval_surfaces": await _aggregate_with_status_counts(
            connection,
            table="memory_retrieval_surfaces",
            user_id=user_id,
            updated_column="updated_at",
        ),
        "contract_dimensions_current": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at
            FROM contract_dimensions_current
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "consequence_chains": await _aggregate_with_status_counts(
            connection,
            table="consequence_chains",
            user_id=user_id,
            updated_column="updated_at",
        ),
        "summary_views": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(created_at) AS max_created_at
            FROM summary_views
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "user_communication_profiles": await _aggregate_profile_markers(
            connection,
            user_id=user_id,
        ),
        "memory_consent_profile": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at
            FROM memory_consent_profile
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "pending_memory_confirmations": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(created_at) AS max_created_at,
                   MAX(asked_at) AS max_asked_at
            FROM pending_memory_confirmations
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "artifacts": await _aggregate_with_status_counts(
            connection,
            table="artifacts",
            user_id=user_id,
            updated_column="updated_at",
        ),
        "artifact_chunks": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at
            FROM artifact_chunks
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "artifact_payload_blobs": await _aggregate_with_status_counts(
            connection,
            table="artifact_payload_blobs",
            user_id=user_id,
            updated_column="updated_at",
        ),
        "artifact_links": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(created_at) AS max_created_at
            FROM artifact_links
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "verbatim_pins": await _aggregate_with_status_counts(
            connection,
            table="verbatim_pins",
            user_id=user_id,
            updated_column="updated_at",
        ),
        "memory_support_edges": await _aggregate_with_status_counts(
            connection,
            table="memory_support_edges",
            user_id=user_id,
            updated_column="updated_at",
        ),
        "memory_evidence_spans": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at,
                   MAX(created_at) AS max_created_at
            FROM memory_evidence_spans
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "graph_entities": await _aggregate_with_status_counts(
            connection,
            table="graph_entities",
            user_id=user_id,
            updated_column="updated_at",
        ),
        "graph_entity_mentions": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(created_at) AS max_created_at
            FROM graph_entity_mentions
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "graph_relationships": await _aggregate_with_status_counts(
            connection,
            table="graph_relationships",
            user_id=user_id,
            updated_column="updated_at",
        ),
        "graph_relationship_sources": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(created_at) AS max_created_at
            FROM graph_relationship_sources
            WHERE user_id = ?
            """,
            (user_id,),
        ),
        "conversation_activity_stats": await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at,
                   MAX(last_message_at) AS max_last_message_at
            FROM conversation_activity_stats
            WHERE user_id = ?
            """,
            (user_id,),
        ),
    }
    if conversation_id is not None:
        sources["conversation"] = await _single_row_marker(
            connection,
            """
            SELECT status,
                   updated_at,
                   last_activity_at,
                   closed_at,
                   temporary,
                   purge_on_close,
                   isolated_mode,
                   incognito,
                   active_presence_id,
                   active_space_id,
                   active_mind_id,
                   active_embodiment_id,
                   active_realm_id
            FROM conversations
            WHERE user_id = ?
              AND id = ?
            """,
            (user_id, conversation_id),
        )
        sources["messages"] = await _aggregate_message_markers(
            connection,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        sources["memory_objects_conversation"] = await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at,
                   MAX(tension_updated_at) AS max_tension_updated_at
            FROM memory_objects
            WHERE user_id = ?
              AND (
                    conversation_id = ?
                 OR archived_by_conversation_id = ?
              )
            """,
            (user_id, conversation_id, conversation_id),
        )
        sources["summary_views_conversation"] = await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(created_at) AS max_created_at
            FROM summary_views
            WHERE user_id = ?
              AND conversation_id = ?
            """,
            (user_id, conversation_id),
        )
        sources["conversation_topics"] = await _aggregate_with_status_counts(
            connection,
            table="conversation_topics",
            user_id=user_id,
            updated_column="updated_at",
            extra_where="conversation_id = ?",
            extra_params=(conversation_id,),
        )
        sources["conversation_topic_events"] = await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(created_at) AS max_created_at
            FROM conversation_topic_events
            WHERE user_id = ?
              AND conversation_id = ?
            """,
            (user_id, conversation_id),
        )
        sources["conversation_topic_sources"] = await _aggregate_one(
            connection,
            """
            SELECT COUNT(*) AS row_count,
                   MAX(created_at) AS max_created_at
            FROM conversation_topic_sources
            WHERE user_id = ?
              AND topic_id IN (
                  SELECT id
                  FROM conversation_topics
                  WHERE user_id = ?
                    AND conversation_id = ?
              )
            """,
            (user_id, user_id, conversation_id),
        )

    markers = {
        "version": 1,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "sources": sources,
    }
    return InitialContextPackageSourceFingerprint(
        source_fingerprint_hash=canonical_json_hash(markers),
        source_markers_json=markers,
    )


async def invalidate_initial_context_package_dependency(
    connection: aiosqlite.Connection,
    *,
    clock: Clock,
    storage_backend: StorageBackend,
    database_path: str,
    user_id: str,
    conversation_id: str | None = None,
    package_kind: InitialContextPackageKind | str | None = None,
    retrieval_profile_id: str | None = None,
    commit: bool = True,
) -> InitialContextPackageCacheInvalidationResult:
    """Mark package rows stale and invalidate dependent context-cache entries."""

    repository = InitialContextPackageRepository(connection, clock)
    if (
        conversation_id is None
        and package_kind is None
        and retrieval_profile_id is None
    ):
        stale_count = await repository.mark_stale_for_user(
            user_id,
            commit=commit,
        )
    else:
        stale_count = await repository.mark_stale_for_key_family(
            user_id=user_id,
            conversation_id=conversation_id,
            package_kind=package_kind,
            retrieval_profile_id=retrieval_profile_id,
            commit=commit,
        )
    if conversation_id is not None:
        deleted_context_views = await storage_backend.delete_context_views_for_conversation(
            user_id,
            conversation_id,
        )
        deleted_recent_windows = await storage_backend.delete_recent_window_for_conversation(
            user_id,
            conversation_id,
        )
    else:
        deleted_context_views = await storage_backend.delete_context_views_for_user(user_id)
        deleted_recent_windows = await storage_backend.delete_recent_windows_for_user(
            user_id
        )
    cache_generation = await storage_backend.increment_cache_generation(
        cache_generation_key(database_path, user_id)
    )
    return InitialContextPackageCacheInvalidationResult(
        stale_package_count=stale_count,
        deleted_context_views=deleted_context_views,
        deleted_recent_windows=deleted_recent_windows,
        cache_generation=cache_generation,
    )


async def _resolve_conversation(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    conversation_id: str | None,
    conversation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if conversation is not None:
        row = dict(conversation)
        if str(row.get("user_id")) != user_id:
            raise ValueError("conversation must belong to user_id")
        if conversation_id is not None and str(row.get("id")) != conversation_id:
            raise ValueError("conversation_id must match conversation")
        return _decode_json_columns(row)
    if conversation_id is None:
        return None
    return await _single_row_marker(
        connection,
        """
        SELECT *
        FROM conversations
        WHERE user_id = ?
          AND id = ?
        """,
        (user_id, conversation_id),
    )


def _conversation_marker(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "workspace_id",
        "assistant_mode_id",
        "status",
        "updated_at",
        "last_activity_at",
        "closed_at",
        "temporary",
        "purge_on_close",
        "isolated_mode",
        "user_persona_id",
        "platform_id",
        "character_id",
        "mode",
        "incognito",
        "active_presence_id",
        "active_space_id",
        "active_mind_id",
        "mind_topology",
        "active_embodiment_id",
        "active_realm_id",
    )
    return {field: _json_safe(row.get(field)) for field in fields}


async def _fetch_user_preferences(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
) -> dict[str, Any] | None:
    return await _single_row_marker(
        connection,
        """
        SELECT id,
               updated_at,
               deleted_at,
               remember_across_chats,
               remember_across_devices,
               memory_privacy_mode
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    )


async def _coordinate_row(
    connection: aiosqlite.Connection,
    *,
    table: str,
    user_id: str,
    id_value: str | None,
    fields: tuple[str, ...],
) -> dict[str, Any] | None:
    if id_value is None:
        return None
    cursor = await connection.execute(
        f"""
        SELECT {", ".join(fields)}
        FROM {table}
        WHERE owner_user_id = ?
          AND id = ?
        """,
        (user_id, id_value),
    )
    row = await cursor.fetchone()
    await cursor.close()
    decoded = _decode_json_columns(row)
    if decoded is None:
        return None
    return {field: _json_safe(decoded.get(field)) for field in fields}


async def _fetch_overseer_grants(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    active_mind_id: str | None,
    mind_topology: str | None,
    now: datetime,
) -> list[dict[str, Any]]:
    overseer_mind_id = _active_overseer_mind_id(
        active_mind_id=active_mind_id,
        mind_topology=mind_topology,
    )
    if overseer_mind_id is None:
        return []
    rows = await _fetch_all(
        connection,
        """
        SELECT owner_user_id,
               overseer_mind_id,
               target_kind,
               target_id,
               grant_kind,
               visibility,
               expires_at,
               revoked_at,
               metadata_json,
               created_at,
               updated_at
        FROM overseer_grants
        WHERE owner_user_id = ?
          AND overseer_mind_id = ?
        ORDER BY target_kind ASC, target_id ASC, grant_kind ASC
        """,
        (user_id, overseer_mind_id),
    )
    markers: list[dict[str, Any]] = []
    for row in rows:
        expires_at = _optional_str(row.get("expires_at"))
        revoked_at = _optional_str(row.get("revoked_at"))
        markers.append(
            {
                **{key: _json_safe(value) for key, value in row.items()},
                "expired": _is_expired(expires_at, now),
                "revoked": revoked_at is not None,
            }
        )
    return markers


async def _fetch_realm_bridges(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    active_realm_id: str | None,
) -> list[dict[str, Any]]:
    if active_realm_id is None:
        return []
    return await _fetch_all(
        connection,
        """
        SELECT owner_user_id,
               source_realm_id,
               target_realm_id,
               cross_realm_mode,
               metadata_json,
               created_at,
               updated_at
        FROM realm_bridges
        WHERE owner_user_id = ?
          AND (
                source_realm_id = ?
             OR target_realm_id = ?
          )
        ORDER BY source_realm_id ASC, target_realm_id ASC
        """,
        (user_id, active_realm_id, active_realm_id),
    )


async def _aggregate_one(
    connection: aiosqlite.Connection,
    query: str,
    parameters: tuple[Any, ...],
) -> dict[str, Any]:
    row = await _single_row_marker(connection, query, parameters)
    return {} if row is None else row


async def _aggregate_joined_to_memory(
    connection: aiosqlite.Connection,
    query: str,
    parameters: tuple[Any, ...],
) -> dict[str, Any]:
    return await _aggregate_one(connection, query, parameters)


async def _aggregate_memory_objects(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
) -> dict[str, Any]:
    aggregate = await _aggregate_one(
        connection,
        """
        SELECT COUNT(*) AS row_count,
               MAX(updated_at) AS max_updated_at,
               MAX(tension_updated_at) AS max_tension_updated_at
        FROM memory_objects
        WHERE user_id = ?
        """,
        (user_id,),
    )
    aggregate["status_counts"] = await _fetch_all(
        connection,
        """
        SELECT status,
               COUNT(*) AS row_count,
               MAX(updated_at) AS max_updated_at,
               MAX(tension_updated_at) AS max_tension_updated_at
        FROM memory_objects
        WHERE user_id = ?
        GROUP BY status
        ORDER BY status ASC
        """,
        (user_id,),
    )
    return aggregate


async def _aggregate_message_markers(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    conversation_id: str,
) -> dict[str, Any]:
    aggregate = await _aggregate_one(
        connection,
        """
        SELECT COUNT(*) AS row_count,
               MAX(seq) AS max_seq,
               MAX(created_at) AS max_created_at,
               MAX(COALESCE(occurred_at, created_at)) AS max_occurred_at,
               SUM(CASE WHEN include_raw = 0 THEN 1 ELSE 0 END) AS hidden_raw_count,
               SUM(CASE WHEN skip_by_default = 1 THEN 1 ELSE 0 END) AS skipped_by_default_count,
               SUM(CASE WHEN heavy_content = 1 THEN 1 ELSE 0 END) AS heavy_content_count,
               SUM(CASE WHEN artifact_backed = 1 THEN 1 ELSE 0 END) AS artifact_backed_count,
               SUM(CASE WHEN verbatim_required = 1 THEN 1 ELSE 0 END) AS verbatim_required_count,
               SUM(CASE WHEN requires_explicit_request = 1 THEN 1 ELSE 0 END) AS requires_explicit_request_count,
               COUNT(DISTINCT content_kind) AS content_kind_count,
               COUNT(DISTINCT COALESCE(context_placeholder, '')) AS context_placeholder_marker_count,
               COUNT(DISTINCT COALESCE(policy_reason, '')) AS policy_reason_marker_count
        FROM messages
        WHERE conversation_id = ?
          AND EXISTS (
              SELECT 1
              FROM conversations
              WHERE conversations.id = messages.conversation_id
                AND conversations.user_id = ?
          )
        """,
        (conversation_id, user_id),
    )
    raw_policy_rows = await _fetch_all(
        connection,
        """
        SELECT id,
               seq,
               content_kind,
               include_raw,
               skip_by_default,
               heavy_content,
               artifact_backed,
               verbatim_required,
               requires_explicit_request,
               context_placeholder,
               policy_reason
        FROM messages
        WHERE conversation_id = ?
          AND EXISTS (
              SELECT 1
              FROM conversations
              WHERE conversations.id = messages.conversation_id
                AND conversations.user_id = ?
          )
        ORDER BY seq ASC, id ASC
        """,
        (conversation_id, user_id),
    )
    buckets: dict[str, dict[str, Any]] = {}
    for row in raw_policy_rows:
        marker_hash = canonical_json_hash(
            {
                "content_kind": row.get("content_kind"),
                "include_raw": row.get("include_raw"),
                "skip_by_default": row.get("skip_by_default"),
                "heavy_content": row.get("heavy_content"),
                "artifact_backed": row.get("artifact_backed"),
                "verbatim_required": row.get("verbatim_required"),
                "requires_explicit_request": row.get("requires_explicit_request"),
                "context_placeholder": row.get("context_placeholder"),
                "policy_reason": row.get("policy_reason"),
            }
        )
        bucket = buckets.setdefault(
            marker_hash,
            {
                "marker_hash": marker_hash,
                "members": [],
            },
        )
        bucket["members"].append(
            {
                "id": row.get("id"),
                "seq": row.get("seq"),
            }
        )
    raw_policy_buckets: list[dict[str, Any]] = []
    for bucket in sorted(buckets.values(), key=lambda item: str(item["marker_hash"])):
        members = sorted(
            bucket["members"],
            key=lambda item: (int(item["seq"]), str(item["id"])),
        )
        seqs = [int(member["seq"]) for member in members]
        raw_policy_buckets.append(
            {
                "marker_hash": bucket["marker_hash"],
                "member_hash": canonical_json_hash({"members": members}),
                "row_count": len(members),
                "min_seq": min(seqs) if seqs else None,
                "max_seq": max(seqs) if seqs else None,
            }
        )
    aggregate["raw_policy_buckets"] = raw_policy_buckets
    return aggregate


async def _aggregate_with_status_counts(
    connection: aiosqlite.Connection,
    *,
    table: str,
    user_id: str,
    updated_column: str,
    extra_where: str | None = None,
    extra_params: tuple[Any, ...] = (),
) -> dict[str, Any]:
    where = "user_id = ?"
    if extra_where is not None:
        where = f"{where} AND {extra_where}"
    aggregate = await _aggregate_one(
        connection,
        f"""
        SELECT COUNT(*) AS row_count,
               MAX({updated_column}) AS max_updated_at
        FROM {table}
        WHERE {where}
        """,
        (user_id, *extra_params),
    )
    status_counts = await _fetch_all(
        connection,
        f"""
        SELECT status,
               COUNT(*) AS row_count,
               MAX({updated_column}) AS max_updated_at
        FROM {table}
        WHERE {where}
        GROUP BY status
        ORDER BY status ASC
        """,
        (user_id, *extra_params),
    )
    aggregate["status_counts"] = status_counts
    return aggregate


async def _aggregate_profile_markers(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
) -> dict[str, Any]:
    aggregate = await _aggregate_one(
        connection,
        """
        SELECT COUNT(*) AS row_count,
               MAX(updated_at) AS max_updated_at,
               SUM(CASE WHEN stale = 1 THEN 1 ELSE 0 END) AS stale_count
        FROM user_communication_profiles
        WHERE user_id = ?
        """,
        (user_id,),
    )
    aggregate["status_counts"] = await _fetch_all(
        connection,
        """
        SELECT status,
               stale,
               COUNT(*) AS row_count,
               MAX(updated_at) AS max_updated_at
        FROM user_communication_profiles
        WHERE user_id = ?
        GROUP BY status, stale
        ORDER BY status ASC, stale ASC
        """,
        (user_id,),
    )
    return aggregate


async def _single_row_marker(
    connection: aiosqlite.Connection,
    query: str,
    parameters: tuple[Any, ...],
) -> dict[str, Any] | None:
    cursor = await connection.execute(query, parameters)
    row = await cursor.fetchone()
    await cursor.close()
    return _decode_json_columns(row)


async def _fetch_all(
    connection: aiosqlite.Connection,
    query: str,
    parameters: tuple[Any, ...],
) -> list[dict[str, Any]]:
    cursor = await connection.execute(query, parameters)
    rows = await cursor.fetchall()
    await cursor.close()
    decoded_rows: list[dict[str, Any]] = []
    for row in rows:
        decoded = _decode_json_columns(row)
        if decoded is not None:
            decoded_rows.append(decoded)
    return decoded_rows


def _decode_json_columns(
    row: Mapping[str, Any] | aiosqlite.Row | None,
) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for key, value in tuple(payload.items()):
        if key.endswith("_json") and isinstance(value, str):
            payload[key] = json_utils.loads(value)
    return {key: _json_safe(value) for key, value in payload.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _coerce_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return _coerce_aware_utc(parsed)


def _is_expired(expires_at: str | None, now: datetime) -> bool:
    parsed = _parse_timestamp(expires_at)
    if parsed is None:
        return False
    return parsed <= now


def _active_overseer_mind_id(
    *,
    active_mind_id: str | None,
    mind_topology: str | None,
) -> str | None:
    if (mind_topology or "unimind") != "ojocentauri":
        return None
    return active_mind_id or DEFAULT_OVERSEER_MIND_ID


def _next_grant_expiry(grants: list[dict[str, Any]]) -> str | None:
    expiries = [
        str(grant["expires_at"])
        for grant in grants
        if grant.get("expires_at") is not None
        and not bool(grant.get("expired"))
        and not bool(grant.get("revoked"))
    ]
    return min(expiries) if expiries else None


def _coalesce(explicit: Any, fallback: Any) -> Any:
    return fallback if explicit is None else explicit


def _coalesce_bool(explicit: bool | None, fallback: Any) -> bool | None:
    if explicit is not None:
        return bool(explicit)
    if fallback is None:
        return None
    return bool(fallback)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
