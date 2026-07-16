"""Reviewed SQLite source-to-revision map for initial context packages."""

from __future__ import annotations

from enum import Enum

import aiosqlite


class InitialContextPackageSourceScope(str, Enum):
    """Revision scope advanced by one canonical source family."""

    USER = "user"
    CONVERSATION = "conversation"


# Keep this map aligned with every canonical table read by the package builder,
# its signature builders, and repositories used to assemble package blocks.
# Policy manifests and operational profiles are intentionally absent: their
# immutable hashes/tokens are part of the package key rather than SQLite state.
INITIAL_CONTEXT_PACKAGE_SOURCE_TO_SCOPE = {
    "users": InitialContextPackageSourceScope.USER,
    "conversations": InitialContextPackageSourceScope.CONVERSATION,
    "messages": InitialContextPackageSourceScope.CONVERSATION,
    "memory_objects": InitialContextPackageSourceScope.USER,
    "belief_versions": InitialContextPackageSourceScope.USER,
    "memory_links": InitialContextPackageSourceScope.USER,
    "memory_retrieval_surfaces": InitialContextPackageSourceScope.USER,
    "contract_dimensions_current": InitialContextPackageSourceScope.USER,
    "consequence_chains": InitialContextPackageSourceScope.USER,
    "summary_views": InitialContextPackageSourceScope.USER,
    "user_communication_profiles": InitialContextPackageSourceScope.USER,
    "memory_consent_profile": InitialContextPackageSourceScope.USER,
    "pending_memory_confirmations": InitialContextPackageSourceScope.USER,
    "artifacts": InitialContextPackageSourceScope.USER,
    "artifact_chunks": InitialContextPackageSourceScope.USER,
    "artifact_payload_blobs": InitialContextPackageSourceScope.USER,
    "artifact_links": InitialContextPackageSourceScope.USER,
    "verbatim_pins": InitialContextPackageSourceScope.USER,
    "memory_support_edges": InitialContextPackageSourceScope.USER,
    "memory_evidence_spans": InitialContextPackageSourceScope.USER,
    "graph_entities": InitialContextPackageSourceScope.USER,
    "graph_entity_mentions": InitialContextPackageSourceScope.USER,
    "graph_relationships": InitialContextPackageSourceScope.USER,
    "graph_relationship_sources": InitialContextPackageSourceScope.USER,
    "conversation_activity_stats": InitialContextPackageSourceScope.CONVERSATION,
    "conversation_topics": InitialContextPackageSourceScope.CONVERSATION,
    "conversation_topic_events": InitialContextPackageSourceScope.CONVERSATION,
    "conversation_topic_sources": InitialContextPackageSourceScope.CONVERSATION,
    "presences": InitialContextPackageSourceScope.USER,
    "memory_object_subjects": InitialContextPackageSourceScope.USER,
    "spaces": InitialContextPackageSourceScope.USER,
    "minds": InitialContextPackageSourceScope.USER,
    "overseer_grants": InitialContextPackageSourceScope.USER,
    "embodiments": InitialContextPackageSourceScope.USER,
    "realms": InitialContextPackageSourceScope.USER,
    "realm_bridges": InitialContextPackageSourceScope.USER,
}


async def assert_initial_context_package_revision_coverage(
    connection: aiosqlite.Connection,
) -> None:
    """Fail when a declared builder input lacks database revision wiring."""

    from atagia.services.initial_context_package_builder import (
        INITIAL_CONTEXT_PACKAGE_SQLITE_SOURCE_TABLES,
    )

    mapped_tables = frozenset(INITIAL_CONTEXT_PACKAGE_SOURCE_TO_SCOPE)
    if mapped_tables != INITIAL_CONTEXT_PACKAGE_SQLITE_SOURCE_TABLES:
        missing_from_map = sorted(
            INITIAL_CONTEXT_PACKAGE_SQLITE_SOURCE_TABLES - mapped_tables
        )
        missing_from_builder = sorted(
            mapped_tables - INITIAL_CONTEXT_PACKAGE_SQLITE_SOURCE_TABLES
        )
        raise RuntimeError(
            "Initial-context-package source declaration mismatch: "
            f"missing_from_map={missing_from_map}, "
            f"missing_from_builder={missing_from_builder}"
        )

    cursor = await connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'trigger'
        """
    )
    trigger_names = {str(row["name"]) for row in await cursor.fetchall()}
    missing: list[str] = []
    for table in sorted(INITIAL_CONTEXT_PACKAGE_SOURCE_TO_SCOPE):
        required_operations = {"au"}
        if table not in {"users", "conversations"}:
            required_operations.update({"ai", "bd"})
        for operation in sorted(required_operations):
            trigger_name = f"icp_source_{table}_{operation}"
            if trigger_name not in trigger_names:
                missing.append(trigger_name)
    if missing:
        raise RuntimeError(
            "Initial-context-package source revision coverage is incomplete: "
            + ", ".join(missing)
        )
