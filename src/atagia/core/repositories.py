"""SQLite repositories for the Step 3 persistence layer."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import math
import re
from typing import Any

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.ids import generate_prefixed_id, new_memory_id
from atagia.core.language_codes import normalize_optional_iso_639_1_code
from atagia.core.storage_backend import StorageBackend
from atagia.core.timestamps import normalize_optional_timestamp
from atagia.models.schemas_memory import (
    ConversationStatus,
    IntimacyBoundary,
    MemoryCategory,
    MindTopology,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    SpaceBoundaryMode,
    SummaryViewKind,
)
from atagia.memory.embodiment_policy import (
    embodiment_visibility_sql_clause_for_context,
)
from atagia.memory.intimacy_boundary_policy import memory_object_intimacy_sql_clause
from atagia.memory.mind_policy import (
    annotate_overseer_grants_for_rows,
    mind_visibility_sql_clause_for_context,
)
from atagia.memory.realm_policy import (
    APPLICABLE_REALM_BRIDGE_MODES,
    annotate_realm_bridge_modes_for_rows,
    realm_visibility_sql_clause_for_context,
)
from atagia.memory.space_policy import space_visibility_sql_clause_for_context
from atagia.services.errors import ConversationNotActiveError, UserDeletedError

RETRIEVAL_ELIGIBLE_MEMORY_STATUSES: tuple[MemoryStatus, ...] = (MemoryStatus.ACTIVE,)
_SAFE_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Categories that the high-risk policy treats as inherently sensitive.
# Used for the strictest-wins sensitivity derivation when callers don't pass
# an explicit value; the namespace redesign plan lists these as private-or-
# stricter regardless of the source row's privacy_level.
_HIGH_RISK_PRIVATE_CATEGORIES: frozenset[MemoryCategory] = frozenset(
    {
        MemoryCategory.MEDICATION,
        MemoryCategory.FINANCIAL,
        MemoryCategory.DATE_OF_BIRTH,
        MemoryCategory.OTHER_SENSITIVE,
    }
)
_HIGH_RISK_SECRET_CATEGORIES: frozenset[MemoryCategory] = frozenset(
    {MemoryCategory.PIN_OR_PASSWORD}
)
# Intimacy boundary values that cannot stay public regardless of the
# privacy_level baseline.
_PRIVATE_INTIMACY_BOUNDARIES: frozenset[IntimacyBoundary] = frozenset(
    {
        IntimacyBoundary.ROMANTIC_PRIVATE,
        IntimacyBoundary.INTIMACY_PRIVATE,
        IntimacyBoundary.INTIMACY_PREFERENCE_PRIVATE,
        IntimacyBoundary.INTIMACY_BOUNDARY,
        IntimacyBoundary.AMBIGUOUS_INTIMATE,
        IntimacyBoundary.SAFETY_BLOCKED,
    }
)

_SENSITIVITY_RANK: dict[MemorySensitivity, int] = {
    MemorySensitivity.UNKNOWN: 0,
    MemorySensitivity.PUBLIC: 1,
    MemorySensitivity.PRIVATE: 2,
    MemorySensitivity.SECRET: 3,
}
_INTIMACY_RANK: dict[IntimacyBoundary, int] = {
    IntimacyBoundary.ORDINARY: 0,
    IntimacyBoundary.ROMANTIC_PRIVATE: 1,
    IntimacyBoundary.INTIMACY_PRIVATE: 2,
    IntimacyBoundary.INTIMACY_PREFERENCE_PRIVATE: 3,
    IntimacyBoundary.INTIMACY_BOUNDARY: 4,
    IntimacyBoundary.AMBIGUOUS_INTIMATE: 5,
    IntimacyBoundary.SAFETY_BLOCKED: 6,
}
_RETRIEVAL_SURFACE_TYPES = frozenset({"pivot", "anchor", "alias", "corpus_surface"})
_RETRIEVAL_SURFACE_STATUSES = frozenset({"active", "stale", "deleted"})
_RETRIEVAL_SURFACE_ANCHOR_TYPES = frozenset(
    {
        "proper_name",
        "person",
        "organization",
        "location",
        "code",
        "quantity",
        "date_time",
        "address",
        "quoted_phrase",
        "attribute",
        "concept",
        "unknown",
    }
)
_RETRIEVAL_SURFACE_ALIAS_KINDS = frozenset(
    {
        "translation",
        "transliteration",
        "spelling_variant",
        "acronym_expansion",
        "domain_synonym",
        "corpus_surface",
    }
)


def _max_sensitivity(*values: MemorySensitivity) -> MemorySensitivity:
    """Return the strictest value (`secret` > `private` > `public` > `unknown`)."""

    return max(values, key=lambda v: _SENSITIVITY_RANK[v])


def _max_intimacy_boundary(*values: IntimacyBoundary) -> IntimacyBoundary:
    """Return the strictest intimacy boundary."""

    return max(values, key=lambda v: _INTIMACY_RANK[v])


def _derive_sensitivity_from_privacy(
    privacy_level: int,
    intimacy_boundary: IntimacyBoundary,
    memory_category: MemoryCategory,
) -> MemorySensitivity:
    """Map legacy fields to the new sensitivity enum with strictest-wins.

    Mirrors the SQL backfill in migration 0031 plus the high-risk and
    intimacy overrides documented in the plan. Used as the default when a
    writer does not pass ``sensitivity`` explicitly.
    """

    if privacy_level >= 3:
        baseline = MemorySensitivity.SECRET
    elif privacy_level == 2:
        baseline = MemorySensitivity.PRIVATE
    elif privacy_level <= 1:
        baseline = MemorySensitivity.PUBLIC
    else:
        baseline = MemorySensitivity.UNKNOWN

    if memory_category in _HIGH_RISK_SECRET_CATEGORIES:
        baseline = _max_sensitivity(baseline, MemorySensitivity.SECRET)
    elif memory_category in _HIGH_RISK_PRIVATE_CATEGORIES:
        baseline = _max_sensitivity(baseline, MemorySensitivity.PRIVATE)
    if intimacy_boundary in _PRIVATE_INTIMACY_BOUNDARIES:
        baseline = _max_sensitivity(baseline, MemorySensitivity.PRIVATE)
    return baseline


def _legacy_scope_to_canonical(scope: MemoryScope) -> str:
    """Map any accepted scope alias to the post-redesign storage scope."""

    if scope is MemoryScope.GLOBAL_USER or scope is MemoryScope.ASSISTANT_MODE or scope is MemoryScope.USER:
        return "user"
    if scope is MemoryScope.CONVERSATION or scope is MemoryScope.EPHEMERAL_SESSION or scope is MemoryScope.CHAT:
        return "chat"
    if scope is MemoryScope.WORKSPACE or scope is MemoryScope.CHARACTER:
        return "character"
    return scope.value


def _decode_json_columns(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for key, value in tuple(payload.items()):
        if key.endswith("_json") and isinstance(value, str):
            payload[key] = json_utils.loads(value)
    return payload


def _encode_json(value: dict[str, Any] | list[Any] | None) -> str:
    if value is None:
        return json_utils.dumps({}, sort_keys=True)
    return json_utils.dumps(value, sort_keys=True)


def _encode_language_codes(language_codes: list[str] | None) -> str | None:
    if not language_codes:
        return None
    normalized_codes: set[str] = set()
    for code in language_codes:
        normalized_code = normalize_optional_iso_639_1_code(code)
        if normalized_code is not None:
            normalized_codes.add(normalized_code)
    normalized = sorted(normalized_codes)
    if not normalized:
        return None
    return json_utils.dumps(normalized)


_HEAVY_MESSAGE_TOKEN_THRESHOLD = 4096


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return default
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_optional_text(value: Any, *, max_length: int | None = None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    if max_length is not None:
        normalized = normalized[:max_length].strip()
    return normalized or None


def _is_mechanically_heavy(text: str, token_count: int | None) -> bool:
    return _message_token_estimate(text, token_count) >= _HEAVY_MESSAGE_TOKEN_THRESHOLD


def _message_token_estimate(text: str, token_count: int | None) -> int:
    if token_count is not None and token_count > 0:
        return token_count
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _build_context_placeholder(
    *,
    message_id: str,
    seq: int | None,
    role: str,
    content_kind: str,
    policy_reason: str,
) -> str:
    seq_part = str(seq) if seq is not None else "?"
    return (
        f"[Skipped message | id={message_id} seq={seq_part} role={role} "
        f"kind={content_kind} policy={policy_reason} ref={message_id}]"
    )


def _stable_string_union(existing: list[str], new_values: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *new_values]:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def summary_mirror_id(summary_view_id: str) -> str:
    """Return the deterministic mirror memory ID for a summary view."""
    return f"sum_mem_{summary_view_id}"


def retrieval_surface_key(surface_text: str) -> str:
    """Return the mechanical dedupe key for a retrieval-only surface."""
    return " ".join(surface_text.split()).casefold()


def retrieval_surface_id(
    *,
    user_id: str,
    memory_id: str,
    surface_type: str,
    language_code: str | None,
    anchor_type: str | None,
    alias_kind: str | None,
    surface_key: str,
) -> str:
    """Return a deterministic id for a memory retrieval surface signature."""
    signature = "\x1f".join(
        [
            user_id,
            memory_id,
            surface_type,
            language_code or "",
            anchor_type or "",
            alias_kind or "",
            surface_key,
        ]
    )
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:32]
    return f"mrs_{digest}"


def _retrieval_surface_nullable_key(value: str | None) -> str:
    if value is None:
        return "n"
    return f"v:{len(value)}:{value}"


def _normalize_required_surface_text(value: str) -> str:
    normalized = " ".join(str(value).split())
    if not normalized:
        raise ValueError("surface_text must be non-empty")
    return normalized


def _normalize_retrieval_surface_type(value: str) -> str:
    normalized = str(value).strip()
    if normalized not in _RETRIEVAL_SURFACE_TYPES:
        raise ValueError(f"Unsupported retrieval surface type: {value!r}")
    return normalized


def _normalize_retrieval_surface_status(value: str) -> str:
    normalized = str(value).strip()
    if normalized not in _RETRIEVAL_SURFACE_STATUSES:
        raise ValueError(f"Unsupported retrieval surface status: {value!r}")
    return normalized


def _normalize_retrieval_anchor_type(value: str | None) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    if normalized not in _RETRIEVAL_SURFACE_ANCHOR_TYPES:
        raise ValueError(f"Unsupported retrieval anchor type: {value!r}")
    return normalized


def _normalize_retrieval_alias_kind(value: str | None) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    if normalized not in _RETRIEVAL_SURFACE_ALIAS_KINDS:
        raise ValueError(f"Unsupported retrieval alias kind: {value!r}")
    return normalized


def _status_filter_clause(
    column_name: str,
    statuses: tuple[MemoryStatus, ...] | None,
) -> tuple[str, list[Any]]:
    if statuses is None:
        return "", []
    placeholders = ", ".join("?" for _ in statuses)
    return f"{column_name} IN ({placeholders})", [status.value for status in statuses]


def conversation_visibility_clause(
    table_alias: str,
    *,
    conversation_id_col: str = "conversation_id",
    include_archived: bool = False,
) -> str:
    """Return a SQL predicate hiding isolated/lifecycle-hidden conversation rows.

    The clause uses one parameter: the active conversation id. Passing ``NULL``
    hides all temporary and isolated conversation rows.
    """
    if not _SAFE_SQL_IDENTIFIER.match(table_alias):
        raise ValueError(f"Unsafe SQL table alias: {table_alias!r}")
    if not _SAFE_SQL_IDENTIFIER.match(conversation_id_col):
        raise ValueError(f"Unsafe SQL column name: {conversation_id_col!r}")
    hidden_status_clause = "visibility_conv.status = 'pending_deletion'"
    if not include_archived:
        hidden_status_clause += " OR visibility_conv.status = 'archived'"
    qualified_column = f"{table_alias}.{conversation_id_col}"
    return (
        "("
        f"{qualified_column} IS NULL "
        f"OR {qualified_column} = ? "
        "OR NOT EXISTS ("
        "SELECT 1 FROM conversations AS visibility_conv "
        f"WHERE visibility_conv.id = {qualified_column} "
        f"AND (visibility_conv.temporary = 1 OR visibility_conv.isolated_mode = 1 OR {hidden_status_clause})"
        ")"
        ")"
    )


class BaseRepository:
    """Shared helpers for SQLite-backed repositories."""

    def __init__(self, connection: aiosqlite.Connection, clock: Clock) -> None:
        self._connection = connection
        self._clock = clock

    def _timestamp(self) -> str:
        return self._clock.now().isoformat()

    async def begin(self) -> None:
        await self._connection.execute("BEGIN")

    async def commit(self) -> None:
        await self._connection.commit()

    async def rollback(self) -> None:
        await self._connection.rollback()

    async def _fetch_one(self, query: str, parameters: tuple[Any, ...]) -> dict[str, Any] | None:
        cursor = await self._connection.execute(query, parameters)
        row = await cursor.fetchone()
        return _decode_json_columns(row)

    async def _fetch_all(self, query: str, parameters: tuple[Any, ...]) -> list[dict[str, Any]]:
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        return [_decode_json_columns(row) for row in rows]


class MemoryRetrievalSurfaceRepository(BaseRepository):
    """Persistence for non-evidential memory retrieval surfaces."""

    async def upsert_surface(
        self,
        *,
        user_id: str,
        memory_id: str,
        surface_type: str,
        surface_text: str,
        anchor_type: str | None = None,
        alias_kind: str | None = None,
        language_code: str | None = None,
        preserve_verbatim: bool = False,
        non_evidential: bool = True,
        confidence: float = 0.5,
        derivation_kind: str = "manual_fixture",
        derivation_model: str | None = None,
        derivation_prompt_version: str | None = None,
        derivation: dict[str, Any] | None = None,
        status: str = "active",
        commit: bool = True,
    ) -> dict[str, Any]:
        if non_evidential is not True:
            raise ValueError("retrieval surfaces must be non_evidential")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        normalized_surface_text = _normalize_required_surface_text(surface_text)
        normalized_surface_type = _normalize_retrieval_surface_type(surface_type)
        normalized_anchor_type = _normalize_retrieval_anchor_type(anchor_type)
        normalized_alias_kind = _normalize_retrieval_alias_kind(alias_kind)
        normalized_language_code = _normalize_optional_text(language_code)
        if normalized_language_code is not None:
            normalized_language_code = normalized_language_code.lower()
        normalized_derivation_kind = _normalize_optional_text(derivation_kind)
        if normalized_derivation_kind is None:
            raise ValueError("derivation_kind must be non-empty")
        normalized_status = _normalize_retrieval_surface_status(status)

        memory = await self._fetch_one(
            """
            SELECT id, user_id
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
            """,
            (memory_id, user_id),
        )
        if memory is None:
            raise ValueError("memory_id must belong to user_id")

        surface_key = retrieval_surface_key(normalized_surface_text)
        surface_id = retrieval_surface_id(
            user_id=user_id,
            memory_id=memory_id,
            surface_type=normalized_surface_type,
            language_code=normalized_language_code,
            anchor_type=normalized_anchor_type,
            alias_kind=normalized_alias_kind,
            surface_key=surface_key,
        )
        timestamp = self._timestamp()
        existing = await self._fetch_one(
            """
            SELECT *
            FROM memory_retrieval_surfaces
            WHERE user_id = ?
              AND memory_id = ?
              AND surface_type = ?
              AND language_key = ?
              AND anchor_type_key = ?
              AND alias_kind_key = ?
              AND surface_key = ?
            """,
            (
                user_id,
                memory_id,
                normalized_surface_type,
                _retrieval_surface_nullable_key(normalized_language_code),
                _retrieval_surface_nullable_key(normalized_anchor_type),
                _retrieval_surface_nullable_key(normalized_alias_kind),
                surface_key,
            ),
        )
        if existing is None:
            await self._connection.execute(
                """
                INSERT INTO memory_retrieval_surfaces(
                    id,
                    user_id,
                    memory_id,
                    surface_type,
                    anchor_type,
                    alias_kind,
                    language_code,
                    surface_text,
                    surface_key,
                    preserve_verbatim,
                    non_evidential,
                    confidence,
                    derivation_kind,
                    derivation_model,
                    derivation_prompt_version,
                    derivation_json,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    surface_id,
                    user_id,
                    memory_id,
                    normalized_surface_type,
                    normalized_anchor_type,
                    normalized_alias_kind,
                    normalized_language_code,
                    normalized_surface_text,
                    surface_key,
                    int(preserve_verbatim),
                    float(confidence),
                    normalized_derivation_kind,
                    _normalize_optional_text(derivation_model),
                    _normalize_optional_text(derivation_prompt_version),
                    _encode_json(derivation or {}),
                    normalized_status,
                    timestamp,
                    timestamp,
                ),
            )
        else:
            surface_id = str(existing["id"])
            await self._connection.execute(
                """
                UPDATE memory_retrieval_surfaces
                SET surface_text = ?,
                    preserve_verbatim = ?,
                    non_evidential = 1,
                    confidence = ?,
                    derivation_kind = ?,
                    derivation_model = ?,
                    derivation_prompt_version = ?,
                    derivation_json = ?,
                    status = ?,
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                """,
                (
                    normalized_surface_text,
                    int(preserve_verbatim),
                    float(confidence),
                    normalized_derivation_kind,
                    _normalize_optional_text(derivation_model),
                    _normalize_optional_text(derivation_prompt_version),
                    _encode_json(derivation or {}),
                    normalized_status,
                    timestamp,
                    surface_id,
                    user_id,
                ),
            )
        if commit:
            await self._connection.commit()
        refreshed = await self.get_surface(surface_id=surface_id, user_id=user_id)
        if refreshed is None:
            raise RuntimeError(f"Failed to upsert retrieval surface {surface_id}")
        return refreshed

    async def get_surface(
        self,
        *,
        surface_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_retrieval_surfaces
            WHERE id = ?
              AND user_id = ?
            """,
            (surface_id, user_id),
        )

    async def list_surfaces_for_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_retrieval_surfaces
            WHERE user_id = ?
              AND memory_id = ?
            ORDER BY surface_type ASC, surface_key ASC, id ASC
            """,
            (user_id, memory_id),
        )

    async def mark_surfaces_stale_for_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
        commit: bool = True,
    ) -> int:
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE memory_retrieval_surfaces
            SET status = 'stale',
                updated_at = ?
            WHERE user_id = ?
              AND memory_id = ?
              AND status != 'deleted'
            """,
            (timestamp, user_id, memory_id),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def mark_surfaces_deleted_for_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
        commit: bool = True,
    ) -> int:
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE memory_retrieval_surfaces
            SET status = 'deleted',
                updated_at = ?
            WHERE user_id = ?
              AND memory_id = ?
            """,
            (timestamp, user_id, memory_id),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def delete_surfaces_for_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
        commit: bool = True,
    ) -> int:
        cursor = await self._connection.execute(
            """
            DELETE FROM memory_retrieval_surfaces
            WHERE user_id = ?
              AND memory_id = ?
            """,
            (user_id, memory_id),
        )
        if commit:
            await self._connection.commit()
        return int(cursor.rowcount or 0)

    async def search_active_surfaces(
        self,
        *,
        user_id: str,
        fts_query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return await self._fetch_all(
            """
            SELECT
                mrs.*,
                mo.status AS memory_status,
                mo.privacy_level AS memory_privacy_level
            FROM memory_retrieval_surfaces_fts
            JOIN memory_retrieval_surfaces AS mrs
              ON mrs._rowid = memory_retrieval_surfaces_fts.rowid
            JOIN memory_objects AS mo
              ON mo.id = mrs.memory_id
            WHERE mrs.user_id = ?
              AND mo.user_id = ?
              AND mrs.status = 'active'
              AND mo.status = ?
              AND memory_retrieval_surfaces_fts MATCH ?
            ORDER BY mrs.updated_at DESC, mrs.id ASC
            LIMIT ?
            """,
            (user_id, user_id, MemoryStatus.ACTIVE.value, fts_query, limit),
        )


class UserRepository(BaseRepository):
    """Persistence operations for users."""

    async def create_user(self, user_id: str | None = None, external_ref: str | None = None) -> dict[str, Any]:
        resolved_user_id = user_id or generate_prefixed_id("usr")
        if await self.has_user_erasure_marker(resolved_user_id):
            raise UserDeletedError("User has been erased")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO users(id, external_ref, created_at, updated_at, deleted_at)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (resolved_user_id, external_ref, timestamp, timestamp),
        )
        await self._connection.commit()
        return await self.get_user(resolved_user_id)

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        )

    async def get_active_user(self, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM users
            WHERE id = ?
              AND deleted_at IS NULL
            """,
            (user_id,),
        )

    async def has_user_erasure_marker(self, user_id: str) -> bool:
        marker_hash = user_erasure_marker_hash(user_id)
        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM deletion_tombstones
            WHERE entity_type = 'user'
              AND deletion_reason = 'right_to_erasure'
              AND json_extract(scope_summary, '$.user_id_sha256') = ?
            LIMIT 1
            """,
            (marker_hash,),
        )
        return await cursor.fetchone() is not None

    async def delete_user(self, user_id: str) -> None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE users
            SET deleted_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (timestamp, timestamp, user_id),
        )
        await self._connection.commit()

    # ------------------------------------------------------------------
    # Namespace redesign: cross-chat / cross-device memory preferences.
    # ------------------------------------------------------------------

    async def get_memory_preferences(
        self, user_id: str
    ) -> dict[str, Any] | None:
        """Return resolved memory preferences for the user.

        Defaults are baked into the schema (both flags start at 1) so a
        legacy row migrated by 0031 returns the same values a fresh row
        would.
        """

        row = await self._fetch_one(
            """
            SELECT remember_across_chats,
                   remember_across_devices,
                   memory_privacy_mode
            FROM users
            WHERE id = ?
              AND deleted_at IS NULL
            """,
            (user_id,),
        )
        if row is None:
            return None
        return {
            "remember_across_chats": bool(row["remember_across_chats"]),
            "remember_across_devices": bool(row["remember_across_devices"]),
            "memory_privacy_mode": str(row["memory_privacy_mode"] or "balanced"),
        }

    async def update_memory_preferences(
        self,
        user_id: str,
        *,
        remember_across_chats: bool | None = None,
        remember_across_devices: bool | None = None,
        memory_privacy_mode: str | None = None,
    ) -> dict[str, Any] | None:
        """Update one or both memory preferences and bump updated_at.

        Cache invalidation (context cache, recent-window cache, retrieval
        snapshots) is intentionally not handled here. Callers compose this
        repository call inside a higher-level service that owns the
        per-user cache generation guard, because the repository layer does
        not know about cache backends. Phase 4 wires that up.
        """

        if (
            remember_across_chats is None
            and remember_across_devices is None
            and memory_privacy_mode is None
        ):
            return await self.get_memory_preferences(user_id)
        existing = await self.get_memory_preferences(user_id)
        if existing is None:
            return None
        new_chats = (
            existing["remember_across_chats"]
            if remember_across_chats is None
            else bool(remember_across_chats)
        )
        new_devices = (
            existing["remember_across_devices"]
            if remember_across_devices is None
            else bool(remember_across_devices)
        )
        new_privacy_mode = (
            str(existing["memory_privacy_mode"])
            if memory_privacy_mode is None
            else str(memory_privacy_mode)
        )
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE users
            SET remember_across_chats = ?,
                remember_across_devices = ?,
                memory_privacy_mode = ?,
                updated_at = ?
            WHERE id = ?
              AND deleted_at IS NULL
            """,
            (
                1 if new_chats else 0,
                1 if new_devices else 0,
                new_privacy_mode,
                timestamp,
                user_id,
            ),
        )
        await self._connection.commit()
        return {
            "remember_across_chats": new_chats,
            "remember_across_devices": new_devices,
            "memory_privacy_mode": new_privacy_mode,
        }


def user_erasure_marker_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


class WorkspaceRepository(BaseRepository):
    """Persistence operations for workspaces."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        storage_backend: StorageBackend | None = None,
    ) -> None:
        super().__init__(connection, clock)
        self._storage_backend = storage_backend

    async def create_workspace(
        self,
        workspace_id: str | None,
        user_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_workspace_id = workspace_id or generate_prefixed_id("wrk")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO workspaces(id, user_id, name, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (resolved_workspace_id, user_id, name, _encode_json(metadata), timestamp, timestamp),
        )
        await self._connection.commit()
        return await self.get_workspace(resolved_workspace_id, user_id)

    async def get_workspace(self, workspace_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM workspaces
            WHERE id = ?
              AND user_id = ?
            """,
            (workspace_id, user_id),
        )

    async def list_workspaces(self, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM workspaces
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (user_id,),
        )

    async def delete_workspace(self, workspace_id: str, user_id: str) -> None:
        await self._connection.execute(
            """
            DELETE FROM workspaces
            WHERE id = ?
              AND user_id = ?
            """,
            (workspace_id, user_id),
        )
        await self._connection.commit()
        if self._storage_backend is not None:
            await self._storage_backend.delete_context_views_for_user(user_id)


class ConversationRepository(BaseRepository):
    """Persistence operations for conversations."""

    async def create_conversation(
        self,
        conversation_id: str | None,
        user_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
        title: str | None,
        metadata: dict[str, Any] | None = None,
        temporary: bool = False,
        temporary_ttl_seconds: int | None = None,
        purge_on_close: bool = False,
        isolated_mode: bool = False,
        *,
        # Namespace redesign identity fields. They are optional during the
        # additive phase so existing callers keep working; Phase 4 makes
        # ``platform_id`` required at the API/proxy/sidecar boundary.
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        active_space_id: str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str = MindTopology.UNIMIND,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
    ) -> dict[str, Any]:
        resolved_conversation_id = conversation_id or generate_prefixed_id("cnv")
        timestamp = self._timestamp()
        # Mirror legacy fields onto the canonical columns so the new
        # repository filters in Phase 3+ have the data they need without
        # the caller having to pass it twice. ``incognito`` defaults to
        # ``isolated_mode`` to preserve current behavior; when callers pass
        # ``incognito`` explicitly, the legacy ``isolated_mode`` column is
        # mirrored from it so legacy retrieval that still reads the
        # ``isolated_mode`` column behaves identically until Phase 11.
        resolved_mode = mode if mode is not None else assistant_mode_id
        resolved_character_id = (
            character_id if character_id is not None else workspace_id
        )
        if incognito is None:
            resolved_incognito = int(isolated_mode)
            resolved_isolated_mode = int(isolated_mode)
        else:
            resolved_incognito = int(bool(incognito))
            resolved_isolated_mode = resolved_incognito
        resolved_mind_topology = MindTopology(mind_topology).value
        await self._connection.execute(
            """
            INSERT INTO conversations(
                id,
                user_id,
                workspace_id,
                assistant_mode_id,
                title,
                status,
                temporary,
                temporary_ttl_seconds,
                purge_on_close,
                isolated_mode,
                last_activity_at,
                closed_at,
                metadata_json,
                created_at,
                updated_at,
                user_persona_id,
                platform_id,
                character_id,
                active_presence_id,
                active_space_id,
                active_mind_id,
                mind_topology,
                active_embodiment_id,
                active_realm_id,
                mode,
                incognito
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_conversation_id,
                user_id,
                workspace_id,
                assistant_mode_id,
                title,
                int(temporary),
                temporary_ttl_seconds,
                int(purge_on_close),
                resolved_isolated_mode,
                timestamp,
                _encode_json(metadata),
                timestamp,
                timestamp,
                user_persona_id,
                platform_id,
                resolved_character_id,
                active_presence_id,
                active_space_id,
                active_mind_id,
                resolved_mind_topology,
                active_embodiment_id,
                active_realm_id,
                resolved_mode,
                resolved_incognito,
            ),
        )
        await self._connection.commit()
        return await self.get_conversation(resolved_conversation_id, user_id)

    async def set_active_presence(
        self,
        conversation_id: str,
        user_id: str,
        active_presence_id: str,
    ) -> dict[str, Any] | None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE conversations
            SET active_presence_id = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (active_presence_id, timestamp, conversation_id, user_id),
        )
        await self._connection.commit()
        return await self.get_conversation(conversation_id, user_id)

    async def set_active_space(
        self,
        conversation_id: str,
        user_id: str,
        active_space_id: str,
    ) -> dict[str, Any] | None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE conversations
            SET active_space_id = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (active_space_id, timestamp, conversation_id, user_id),
        )
        await self._connection.commit()
        return await self.get_conversation(conversation_id, user_id)

    async def set_active_mind(
        self,
        conversation_id: str,
        user_id: str,
        active_mind_id: str,
        mind_topology: MindTopology | str,
    ) -> dict[str, Any] | None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE conversations
            SET active_mind_id = ?,
                mind_topology = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                active_mind_id,
                MindTopology(mind_topology).value,
                timestamp,
                conversation_id,
                user_id,
            ),
        )
        await self._connection.commit()
        return await self.get_conversation(conversation_id, user_id)

    async def set_active_embodiment(
        self,
        conversation_id: str,
        user_id: str,
        active_embodiment_id: str,
    ) -> dict[str, Any] | None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE conversations
            SET active_embodiment_id = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (active_embodiment_id, timestamp, conversation_id, user_id),
        )
        await self._connection.commit()
        return await self.get_conversation(conversation_id, user_id)

    async def set_active_realm(
        self,
        conversation_id: str,
        user_id: str,
        active_realm_id: str,
    ) -> dict[str, Any] | None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE conversations
            SET active_realm_id = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (active_realm_id, timestamp, conversation_id, user_id),
        )
        await self._connection.commit()
        return await self.get_conversation(conversation_id, user_id)

    async def get_conversation(self, conversation_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM conversations
            WHERE id = ?
              AND user_id = ?
            """,
            (conversation_id, user_id),
        )

    async def get_active_conversation(self, conversation_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM conversations
            WHERE id = ?
              AND user_id = ?
              AND status = ?
            """,
            (conversation_id, user_id, ConversationStatus.ACTIVE.value),
        )

    async def list_conversations(
        self,
        user_id: str,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        status: str | None = None,
        include_temporary: bool = False,
        include_archived: bool = False,
        *,
        namespace_filter: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            parameters.append(workspace_id)
        if assistant_mode_id is not None:
            clauses.append("assistant_mode_id = ?")
            parameters.append(assistant_mode_id)
        if namespace_filter:
            clauses.append("user_persona_id IS ?")
            clauses.append("platform_id = ?")
            clauses.append("character_id IS ?")
            clauses.append("incognito = ?")
            parameters.extend([user_persona_id, platform_id, character_id, 1 if incognito else 0])
        if status is not None:
            ConversationStatus(status)
            clauses.append("status = ?")
            parameters.append(status)
        elif include_archived:
            clauses.append("status IN (?, ?, ?)")
            parameters.extend(
                [
                    ConversationStatus.ACTIVE.value,
                    ConversationStatus.CLOSED.value,
                    ConversationStatus.ARCHIVED.value,
                ]
            )
        else:
            clauses.append("status = ?")
            parameters.append(ConversationStatus.ACTIVE.value)
        if not include_temporary:
            clauses.append("temporary = 0")

        return await self._fetch_all(
            """
            SELECT *
            FROM conversations
            WHERE {where_clause}
            ORDER BY updated_at DESC, id ASC
            """.format(where_clause=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def mark_conversation_isolated(self, conversation_id: str, user_id: str) -> dict[str, Any] | None:
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE conversations
            SET isolated_mode = 1,
                incognito = 1,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND isolated_mode = 0
            """,
            (timestamp, conversation_id, user_id),
        )
        await self._connection.commit()
        return await self.get_conversation(conversation_id, user_id)

    async def set_conversation_incognito(
        self,
        conversation_id: str,
        user_id: str,
        incognito: bool,
        *,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        """Toggle the per-conversation ``incognito`` flag.

        Reversible per the redesign plan: setting ``incognito=False`` after
        a previous ``True`` does *not* automatically re-promote derived data
        that was narrowed while incognito was on. Cache invalidation,
        transactional narrowing of broad derived rows, and queued-job
        cancellation belong to the service layer that owns those caches and
        workers (Phase 4 + Phase 8); this method is the underlying flip.
        """

        timestamp = self._timestamp()
        target = 1 if incognito else 0
        await self._connection.execute(
            """
            UPDATE conversations
            SET incognito = ?,
                isolated_mode = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (target, target, timestamp, conversation_id, user_id),
        )
        if commit:
            await self._connection.commit()
        return await self.get_conversation(conversation_id, user_id)

    async def update_conversation_status(self, conversation_id: str, user_id: str, status: str) -> None:
        ConversationStatus(status)
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE conversations
            SET status = ?,
                updated_at = ?,
                closed_at = CASE WHEN ? = 'closed' THEN COALESCE(closed_at, ?) ELSE closed_at END
            WHERE id = ?
              AND user_id = ?
            """,
            (status, timestamp, status, timestamp, conversation_id, user_id),
        )
        await self._connection.commit()


class MessageRepository(BaseRepository):
    """Persistence operations for messages plus Step 3 FTS search."""

    @staticmethod
    def _derive_message_policy(
        *,
        message_id: str,
        role: str,
        seq: int | None,
        text: str,
        token_count: int | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        explicit_content_kind = _normalize_optional_text(metadata.get("content_kind"), max_length=64)
        explicit_include_raw = "include_raw" in metadata
        explicit_policy_reason = _normalize_optional_text(metadata.get("policy_reason"), max_length=128)
        explicit_context_placeholder = _normalize_optional_text(
            metadata.get("context_placeholder"),
            max_length=300,
        )

        mechanical_heavy = _is_mechanically_heavy(text, token_count)
        heavy_content = _coerce_bool(metadata.get("heavy_content")) or mechanical_heavy
        artifact_backed = _coerce_bool(metadata.get("artifact_backed"))
        verbatim_required = _coerce_bool(metadata.get("verbatim_required"))
        skip_by_default = _coerce_bool(metadata.get("skip_by_default"))
        include_raw = _coerce_bool(metadata.get("include_raw"), True)
        requires_explicit_request = _coerce_bool(metadata.get("requires_explicit_request"))

        if heavy_content or artifact_backed or verbatim_required or skip_by_default:
            if not explicit_include_raw:
                include_raw = False
        if not include_raw:
            skip_by_default = True
        if heavy_content or artifact_backed or verbatim_required:
            skip_by_default = True if not explicit_include_raw else skip_by_default

        requires_explicit_request = (
            requires_explicit_request
            or skip_by_default
            or heavy_content
            or artifact_backed
            or verbatim_required
        )
        if explicit_include_raw and include_raw:
            requires_explicit_request = _coerce_bool(metadata.get("requires_explicit_request"))

        if explicit_content_kind is not None:
            content_kind = explicit_content_kind
        elif artifact_backed:
            content_kind = "artifact"
        elif heavy_content or skip_by_default:
            content_kind = "attachment"
        else:
            content_kind = "text"

        policy_reason = explicit_policy_reason
        if policy_reason is None:
            if artifact_backed:
                policy_reason = "artifact_backed"
            elif verbatim_required:
                policy_reason = "verbatim_required"
            elif heavy_content and mechanical_heavy:
                policy_reason = "mechanical_size_threshold"
            elif skip_by_default and not include_raw:
                policy_reason = "skip_by_default"
            elif not include_raw:
                policy_reason = "include_raw_false"

        context_placeholder = explicit_context_placeholder
        if context_placeholder is None and skip_by_default and seq is not None:
            context_placeholder = _build_context_placeholder(
                message_id=message_id,
                seq=seq,
                role=role,
                content_kind=content_kind,
                policy_reason=policy_reason or "skip_by_default",
            )

        return {
            "content_kind": content_kind,
            "include_raw": int(include_raw),
            "skip_by_default": int(skip_by_default),
            "heavy_content": int(heavy_content),
            "artifact_backed": int(artifact_backed),
            "verbatim_required": int(verbatim_required),
            "requires_explicit_request": int(requires_explicit_request),
            "context_placeholder": context_placeholder,
            "policy_reason": policy_reason,
        }

    async def next_sequence(self, conversation_id: str) -> int:
        cursor = await self._connection.execute(
            """
            SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq
            FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return int(row["next_seq"])

    async def create_message(
        self,
        message_id: str | None,
        conversation_id: str,
        role: str,
        seq: int | None,
        text: str,
        token_count: int | None = None,
        metadata: dict[str, Any] | None = None,
        occurred_at: str | None = None,
        *,
        active_presence_id: str | None = None,
        source_presence_id: str | None = None,
        space_id: str | None = None,
        active_mind_id: str | None = None,
        source_mind_id: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_message_id = message_id or generate_prefixed_id("msg")
        timestamp = self._timestamp()
        resolved_occurred_at = normalize_optional_timestamp(occurred_at)
        message_policy = self._derive_message_policy(
            message_id=resolved_message_id,
            role=role,
            seq=seq,
            text=text,
            token_count=token_count,
            metadata=metadata,
        )
        if seq is None:
            cursor = await self._connection.execute(
                """
                INSERT INTO messages(
                    id,
                    conversation_id,
                    role,
                    seq,
                    text,
                    token_count,
                    metadata_json,
                    content_kind,
                    include_raw,
                    skip_by_default,
                    heavy_content,
                    artifact_backed,
                    verbatim_required,
                    requires_explicit_request,
                    context_placeholder,
                    policy_reason,
                    active_presence_id,
                    source_presence_id,
                    space_id,
                    active_mind_id,
                    source_mind_id,
                    active_embodiment_id,
                    active_realm_id,
                    created_at,
                    occurred_at
                )
                SELECT
                    ?,
                    c.id,
                    ?,
                    COALESCE(
                        (
                            SELECT MAX(existing.seq)
                            FROM messages AS existing
                            WHERE existing.conversation_id = c.id
                        ),
                        0
                    ) + 1,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                FROM conversations AS c
                WHERE c.id = ?
                  AND c.status = 'active'
                """,
                (
                    resolved_message_id,
                    role,
                    text,
                    token_count,
                    _encode_json(metadata),
                    message_policy["content_kind"],
                    message_policy["include_raw"],
                    message_policy["skip_by_default"],
                    message_policy["heavy_content"],
                    message_policy["artifact_backed"],
                    message_policy["verbatim_required"],
                    message_policy["requires_explicit_request"],
                    message_policy["context_placeholder"],
                    message_policy["policy_reason"],
                    active_presence_id,
                    source_presence_id,
                    space_id,
                    active_mind_id,
                    source_mind_id,
                    active_embodiment_id,
                    active_realm_id,
                    timestamp,
                    resolved_occurred_at,
                    conversation_id,
                ),
            )
        else:
            cursor = await self._connection.execute(
                """
                INSERT INTO messages(
                    id,
                    conversation_id,
                    role,
                    seq,
                    text,
                    token_count,
                    metadata_json,
                    content_kind,
                    include_raw,
                    skip_by_default,
                    heavy_content,
                    artifact_backed,
                    verbatim_required,
                    requires_explicit_request,
                    context_placeholder,
                    policy_reason,
                    active_presence_id,
                    source_presence_id,
                    space_id,
                    active_mind_id,
                    source_mind_id,
                    active_embodiment_id,
                    active_realm_id,
                    created_at,
                    occurred_at
                )
                SELECT
                    ?,
                    c.id,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                FROM conversations AS c
                WHERE c.id = ?
                  AND c.status = 'active'
                """,
                (
                    resolved_message_id,
                    role,
                    seq,
                    text,
                    token_count,
                    _encode_json(metadata),
                    message_policy["content_kind"],
                    message_policy["include_raw"],
                    message_policy["skip_by_default"],
                    message_policy["heavy_content"],
                    message_policy["artifact_backed"],
                    message_policy["verbatim_required"],
                    message_policy["requires_explicit_request"],
                    message_policy["context_placeholder"],
                    message_policy["policy_reason"],
                    active_presence_id,
                    source_presence_id,
                    space_id,
                    active_mind_id,
                    source_mind_id,
                    active_embodiment_id,
                    active_realm_id,
                    timestamp,
                    resolved_occurred_at,
                    conversation_id,
                ),
            )
        if cursor.rowcount == 0:
            raise ConversationNotActiveError("Conversation is not active")
        await self._connection.execute(
            """
            UPDATE conversations
            SET last_activity_at = CASE
                    WHEN last_activity_at IS NULL THEN ?
                    WHEN datetime(?) > datetime(last_activity_at) THEN ?
                    ELSE last_activity_at
                END,
                updated_at = ?
            WHERE id = ?
              AND status = 'active'
            """,
            (
                resolved_occurred_at or timestamp,
                resolved_occurred_at or timestamp,
                resolved_occurred_at or timestamp,
                timestamp,
                conversation_id,
            ),
        )
        if commit:
            await self._connection.commit()
        created = await self._fetch_one(
            "SELECT * FROM messages WHERE id = ?",
            (resolved_message_id,),
        )
        if created is None:
            raise RuntimeError(f"Failed to create message {resolved_message_id}")
        if seq is None and created.get("skip_by_default"):
            generated_placeholder = _build_context_placeholder(
                message_id=resolved_message_id,
                seq=int(created["seq"]),
                role=role,
                content_kind=str(created.get("content_kind") or "text"),
                policy_reason=str(created.get("policy_reason") or "skip_by_default"),
            )
            if created.get("context_placeholder") != generated_placeholder:
                await self._connection.execute(
                    """
                    UPDATE messages
                    SET context_placeholder = ?
                    WHERE id = ?
                    """,
                    (generated_placeholder, resolved_message_id),
                )
                if commit:
                    await self._connection.commit()
                created["context_placeholder"] = generated_placeholder
        return created

    async def get_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
            ORDER BY m.seq ASC
            LIMIT ?
            OFFSET ?
            """,
            (conversation_id, user_id, limit, offset),
        )

    async def get_recent_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM (
                SELECT m.*
                FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE m.conversation_id = ?
                  AND c.user_id = ?
                ORDER BY m.seq DESC
                LIMIT ?
            ) AS recent_messages
            ORDER BY seq ASC
            """,
            (conversation_id, user_id, limit),
        )

    async def get_recent_messages_before_seq(
        self,
        conversation_id: str,
        user_id: str,
        before_seq: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM (
                SELECT m.*
                FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE m.conversation_id = ?
                  AND c.user_id = ?
                  AND m.seq < ?
                ORDER BY m.seq DESC
                LIMIT ?
            ) AS recent_messages
            ORDER BY seq ASC
            """,
            (conversation_id, user_id, before_seq, limit),
        )

    async def list_messages_for_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
            ORDER BY m.seq ASC
            """,
            (conversation_id, user_id),
        )

    async def get_messages_in_seq_range(
        self,
        conversation_id: str,
        user_id: str,
        start_seq: int,
        end_seq: int,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT
                m.*,
                sp.boundary_mode AS message_space_boundary_mode
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            LEFT JOIN spaces AS sp
              ON sp.owner_user_id = c.user_id
             AND sp.id = m.space_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.seq BETWEEN ? AND ?
            ORDER BY m.seq ASC
            """,
            (conversation_id, user_id, start_seq, end_seq),
        )

    async def get_messages_from_seq(
        self,
        conversation_id: str,
        user_id: str,
        start_seq: int,
    ) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.seq >= ?
            ORDER BY m.seq ASC
            """,
            (conversation_id, user_id, start_seq),
        )

    async def get_message(self, message_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.id = ?
              AND c.user_id = ?
            """,
            (message_id, user_id),
        )

    async def get_message_by_seq(
        self,
        conversation_id: str,
        user_id: str,
        seq: int,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.seq = ?
            """,
            (conversation_id, user_id, seq),
        )

    async def get_message_for_idempotency(
        self,
        message_id: str,
    ) -> dict[str, Any] | None:
        """Return a message with owner metadata for internal idempotency checks."""
        return await self._fetch_one(
            """
            SELECT
                m.*,
                c.user_id AS _conversation_user_id
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE m.id = ?
            """,
            (message_id,),
        )

    async def search_messages(
        self,
        user_id: str,
        query: str,
        limit: int,
        *,
        allow_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        isolate_to_allowed = await self._conversation_is_isolated(allow_conversation_id, user_id)
        return await self._fetch_all(
            """
            SELECT
                m.*,
                bm25(messages_fts) AS rank
            FROM messages_fts
            JOIN messages AS m ON m._rowid = messages_fts.rowid
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE c.user_id = ?
              AND {visibility_clause}
              AND (? = 0 OR c.id = ?)
              AND messages_fts MATCH ?
            ORDER BY rank ASC, m.seq ASC
            LIMIT ?
            """.format(
                visibility_clause=conversation_visibility_clause("c", conversation_id_col="id"),
            ),
            (
                user_id,
                allow_conversation_id,
                int(isolate_to_allowed),
                allow_conversation_id,
                query,
                limit,
            ),
        )

    async def search_messages_with_privacy(
        self,
        *,
        user_id: str,
        query: str,
        privacy_ceiling: int,
        limit: int,
        allow_conversation_id: str | None = None,
        active_conversation_only: bool = False,
        include_pending_confirmation_sources: bool = False,
    ) -> list[dict[str, Any]]:
        """Wave 1 batch 2 (1-C): privacy-filtered FTS search on messages.

        Security invariants enforced at the SQL layer, BEFORE ranking:
        - ``user_id`` filter: messages belong to the caller's conversations.
        - Mode privacy ceiling: conversations whose owning assistant_mode
          has a privacy_ceiling strictly greater than the current
          retrieval ceiling are excluded entirely. Therapy-mode messages
          can never surface inside a general-mode retrieval even though
          they share a user. The active conversation may opt into a
          narrow same-context bypass via ``allow_conversation_id``.
        - Consent gating: messages that are the verbatim source of a
          memory object still waiting for user confirmation are excluded
          so unconfirmed sensitive content never leaks through the raw
          channel.
        """
        if limit <= 0:
            return []
        isolate_to_allowed = await self._conversation_is_isolated(allow_conversation_id, user_id)
        return await self._fetch_all(
            """
            SELECT
                m.*,
                c.assistant_mode_id AS conversation_assistant_mode_id,
                am.privacy_ceiling AS mode_privacy_ceiling,
                sp.boundary_mode AS message_space_boundary_mode,
                bm25(messages_fts) AS rank,
                snippet(messages_fts, 0, '', '', ' ... ', 64) AS fts_snippet
            FROM messages_fts
            JOIN messages AS m ON m._rowid = messages_fts.rowid
            JOIN conversations AS c ON c.id = m.conversation_id
            JOIN assistant_modes AS am ON am.id = c.assistant_mode_id
            LEFT JOIN spaces AS sp
              ON sp.owner_user_id = c.user_id
             AND sp.id = m.space_id
            WHERE c.user_id = ?
              AND (am.privacy_ceiling <= ? OR (? IS NOT NULL AND c.id = ?))
              AND c.status = 'active'
              AND {visibility_clause}
              AND (? = 0 OR c.id = ?)
              AND (? = 0 OR c.id = ?)
              AND (? = 1 OR NOT EXISTS (
                  SELECT 1
                  FROM memory_objects AS pending
                  JOIN json_each(
                      json_extract(pending.payload_json, '$.source_message_ids')
                  ) AS pending_src ON pending_src.value = m.id
                  WHERE pending.user_id = ?
                    AND pending.status = 'pending_user_confirmation'
              ))
              AND messages_fts MATCH ?
            ORDER BY rank ASC, m.seq ASC
            LIMIT ?
            """.format(
                visibility_clause=conversation_visibility_clause("c", conversation_id_col="id"),
            ),
            (
                user_id,
                privacy_ceiling,
                allow_conversation_id,
                allow_conversation_id,
                allow_conversation_id,
                int(isolate_to_allowed),
                allow_conversation_id,
                int(active_conversation_only),
                allow_conversation_id,
                int(include_pending_confirmation_sources),
                user_id,
                query,
                limit,
            ),
        )

    async def _conversation_is_isolated(
        self,
        conversation_id: str | None,
        user_id: str,
    ) -> bool:
        if conversation_id is None:
            return False
        cursor = await self._connection.execute(
            """
            SELECT isolated_mode
            FROM conversations
            WHERE id = ?
              AND user_id = ?
            """,
            (conversation_id, user_id),
        )
        row = await cursor.fetchone()
        return bool(row is not None and row["isolated_mode"])

    async def fetch_message_window(
        self,
        *,
        conversation_id: str,
        user_id: str,
        center_seq: int,
        window_size: int,
    ) -> list[dict[str, Any]]:
        """Fetch up to ``window_size`` contiguous messages around ``center_seq``.

        The window is built by ``seq`` arithmetic only: ``start_seq`` and
        ``end_seq`` are derived from ``center_seq`` and ``window_size``
        without requiring the center row to exist. Messages are returned
        in ascending ``seq`` order so the list reads as a contiguous
        transcript. ``user_id`` is enforced via the owning conversation
        to preserve the per-user isolation invariant.
        """
        if window_size <= 0:
            return []
        half = max(0, (window_size - 1) // 2)
        start_seq = max(0, center_seq - half)
        end_seq = center_seq + (window_size - 1 - half)
        return await self._fetch_all(
            """
            SELECT
                m.*,
                sp.boundary_mode AS message_space_boundary_mode
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            LEFT JOIN spaces AS sp
              ON sp.owner_user_id = c.user_id
             AND sp.id = m.space_id
            WHERE m.conversation_id = ?
              AND c.user_id = ?
              AND m.seq BETWEEN ? AND ?
            ORDER BY m.seq ASC
            """,
            (conversation_id, user_id, start_seq, end_seq),
        )

    async def sum_text_length_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        conversation_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
    ) -> int:
        """Return the total message text length eligible for small-corpus mode.

        Messages inherit their scope from the owning conversation. The widest
        allowed scope determines how many conversations contribute: a
        `global_user` allowance sweeps every user conversation, while
        `assistant_mode`, `workspace`, `conversation`, and `ephemeral_session`
        progressively narrow the set.
        """
        clauses, parameters = self._message_scope_clauses(
            scopes,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
        )
        if not clauses:
            return 0

        cursor = await self._connection.execute(
            """
            SELECT COALESCE(SUM(LENGTH(m.text)), 0) AS total_length
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE c.user_id = ?
              AND (
                  {clauses}
              )
              AND {visibility_clause}
            """.format(
                clauses=" OR ".join(clauses),
                visibility_clause=conversation_visibility_clause("c", conversation_id_col="id"),
            ),
            tuple([user_id, *parameters, conversation_id]),
        )
        row = await cursor.fetchone()
        return int(row["total_length"])

    async def list_eligible_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        conversation_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
    ) -> list[dict[str, Any]]:
        """Return every message eligible for small-corpus composition.

        Messages are ordered by seq within each conversation, and across
        conversations by creation time, so callers get a deterministic
        transcript-shaped feed.
        """
        clauses, parameters = self._message_scope_clauses(
            scopes,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
        )
        if not clauses:
            return []

        return await self._fetch_all(
            """
            SELECT m.*
            FROM messages AS m
            JOIN conversations AS c ON c.id = m.conversation_id
            WHERE c.user_id = ?
              AND (
                  {clauses}
              )
              AND {visibility_clause}
            ORDER BY c.created_at ASC, m.conversation_id ASC, m.seq ASC
            """.format(
                clauses=" OR ".join(clauses),
                visibility_clause=conversation_visibility_clause("c", conversation_id_col="id"),
            ),
            tuple([user_id, *parameters, conversation_id]),
        )

    @staticmethod
    def _message_scope_clauses(
        scopes: list[MemoryScope],
        *,
        conversation_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in scopes:
            if scope is MemoryScope.GLOBAL_USER:
                clauses.append("1 = 1")
            elif scope is MemoryScope.WORKSPACE and workspace_id is not None:
                clauses.append("(c.workspace_id = ?)")
                parameters.append(workspace_id)
            elif scope is MemoryScope.CONVERSATION:
                clauses.append("(c.id = ?)")
                parameters.append(conversation_id)
            elif scope is MemoryScope.EPHEMERAL_SESSION:
                clauses.append("(c.id = ?)")
                parameters.append(conversation_id)
        return clauses, parameters


class MemoryObjectRepository(BaseRepository):
    """Persistence operations for canonical memory objects."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(connection, clock)
        self._settings = settings or Settings.from_env()

    async def _mark_retrieval_surfaces_stale_for_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
        timestamp: str,
    ) -> int:
        cursor = await self._connection.execute(
            """
            UPDATE memory_retrieval_surfaces
            SET status = 'stale',
                updated_at = ?
            WHERE user_id = ?
              AND memory_id = ?
              AND status != 'deleted'
            """,
            (timestamp, user_id, memory_id),
        )
        return int(cursor.rowcount or 0)

    async def _mark_retrieval_surfaces_deleted_for_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
        timestamp: str,
    ) -> int:
        cursor = await self._connection.execute(
            """
            UPDATE memory_retrieval_surfaces
            SET status = 'deleted',
                updated_at = ?
            WHERE user_id = ?
              AND memory_id = ?
            """,
            (timestamp, user_id, memory_id),
        )
        return int(cursor.rowcount or 0)

    async def get_memory_object(self, memory_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
            """,
            (memory_id, user_id),
        )

    async def get_visible_memory_object(
        self,
        memory_id: str,
        user_id: str,
        *,
        conversation_id: str,
        user_persona_id: str | None,
        platform_id: str,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        sensitivity_gates_enabled: bool = False,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a memory only if it is visible in the active namespace."""

        clauses, parameters = self.namespace_visibility_clauses(
            [MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER],
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            conversation_id=conversation_id,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            incognito=incognito,
            sensitivity_gates_enabled=sensitivity_gates_enabled,
            table_alias="mo",
        )
        if not clauses:
            return None
        space_clause, space_parameters = space_visibility_sql_clause_for_context(
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            alias="mo",
        )
        mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            alias="mo",
        )
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
            active_embodiment_id=active_embodiment_id,
            alias="mo",
        )
        realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
            active_realm_id=active_realm_id,
            alias="mo",
        )
        return await self._fetch_one(
            """
            SELECT mo.*
            FROM memory_objects AS mo
            WHERE mo.id = ?
              AND mo.user_id = ?
              AND mo.status != ?
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND {namespace_clauses}
              AND {space_clause}
              AND {mind_clause}
              AND {embodiment_clause}
              AND {realm_clause}
            """.format(
                visibility_clause=conversation_visibility_clause("mo"),
                namespace_clauses=" AND ".join(clauses),
                space_clause=space_clause,
                mind_clause=mind_clause,
                embodiment_clause=embodiment_clause,
                realm_clause=realm_clause,
            ),
            (
                memory_id,
                user_id,
                MemoryStatus.DELETED.value,
                conversation_id,
                *parameters,
                *space_parameters,
                *mind_parameters,
                *embodiment_parameters,
                *realm_parameters,
            ),
        )

    async def archive_memory_object(
        self,
        memory_id: str,
        user_id: str,
        *,
        commit: bool = True,
    ) -> bool:
        """Archive an active memory object and report whether a row was updated."""
        timestamp = self._timestamp()
        cursor = await self._connection.execute(
            """
            UPDATE memory_objects
            SET status = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND status = ?
            """,
            (
                MemoryStatus.ARCHIVED.value,
                timestamp,
                memory_id,
                user_id,
                MemoryStatus.ACTIVE.value,
            ),
        )
        if cursor.rowcount:
            await self._mark_retrieval_surfaces_stale_for_memory(
                user_id=user_id,
                memory_id=memory_id,
                timestamp=timestamp,
            )
        if commit:
            await self._connection.commit()
        return cursor.rowcount > 0

    async def list_memory_objects_by_ids(
        self,
        user_id: str,
        memory_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not memory_ids:
            return []
        placeholders = ", ".join("?" for _ in memory_ids)
        return await self._fetch_all(
            f"""
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND id IN ({placeholders})
            ORDER BY created_at ASC, id ASC
            """,
            (user_id, *memory_ids),
        )

    async def update_memory_object_status(
        self,
        *,
        memory_id: str,
        user_id: str,
        status: MemoryStatus,
        payload_updates: dict[str, Any] | None = None,
        expected_current_status: MemoryStatus | None = None,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        existing = await self.get_memory_object(memory_id, user_id)
        if existing is None:
            return None
        if expected_current_status is not None and existing["status"] != expected_current_status.value:
            return None

        payload = existing.get("payload_json")
        normalized_payload = dict(payload) if isinstance(payload, dict) else {}
        if payload_updates:
            normalized_payload.update(payload_updates)
        timestamp = self._timestamp()
        parameters: list[Any] = [
            status.value,
            _encode_json(normalized_payload),
            timestamp,
            memory_id,
            user_id,
        ]
        status_clause = ""
        if expected_current_status is not None:
            status_clause = " AND status = ?"
            parameters.append(expected_current_status.value)
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET status = ?,
                payload_json = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            {status_clause}
            """.format(status_clause=status_clause),
            tuple(parameters),
        )
        if status is MemoryStatus.DELETED:
            await self._mark_retrieval_surfaces_deleted_for_memory(
                user_id=user_id,
                memory_id=memory_id,
                timestamp=timestamp,
            )
        elif status is not MemoryStatus.ACTIVE:
            await self._mark_retrieval_surfaces_stale_for_memory(
                user_id=user_id,
                memory_id=memory_id,
                timestamp=timestamp,
            )
        if commit:
            await self._connection.commit()
        return await self.get_memory_object(memory_id, user_id)

    async def upsert_summary_mirror(
        self,
        *,
        user_id: str,
        summary_view_id: str,
        summary_kind: SummaryViewKind,
        hierarchy_level: int,
        summary_text: str,
        source_object_ids: list[str],
        created_at: str,
        updated_at: str | None = None,
        index_text: str | None = None,
        scope: MemoryScope = MemoryScope.GLOBAL_USER,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        confidence: float = 0.72,
        stability: float = 0.82,
        vitality: float = 0.15,
        maya_score: float = 1.5,
        privacy_level: int = 0,
        intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        payload: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        scope_canonical: str | None = None,
        language_codes: list[str] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        mirror_id = summary_mirror_id(summary_view_id)
        existing = await self.get_memory_object(mirror_id, user_id)
        language_codes_json = _encode_language_codes(language_codes)
        normalized_source_ids = [
            str(item).strip()
            for item in source_object_ids
            if str(item).strip()
        ]
        normalized_payload = {
            **(payload or {}),
            "summary_view_id": summary_view_id,
            "summary_kind": summary_kind.value,
            "hierarchy_level": hierarchy_level,
            "source_object_ids": normalized_source_ids,
        }

        if existing is None:
            return await self.create_memory_object(
                user_id=user_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                assistant_mode_id=assistant_mode_id,
                object_type=MemoryObjectType.SUMMARY_VIEW,
                scope=scope,
                canonical_text=summary_text,
                index_text=index_text,
                payload=normalized_payload,
                source_kind=MemorySourceKind.SUMMARIZED,
                confidence=confidence,
                stability=stability,
                vitality=vitality,
                maya_score=maya_score,
                privacy_level=privacy_level,
                memory_category=MemoryCategory.UNKNOWN,
                intimacy_boundary=intimacy_boundary,
                intimacy_boundary_confidence=intimacy_boundary_confidence,
                preserve_verbatim=False,
                status=status,
                memory_id=mirror_id,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                sensitivity=sensitivity,
                themes=themes,
                platform_locked=platform_locked,
                platform_id_lock=platform_id_lock,
                scope_canonical=scope_canonical,
                language_codes=language_codes,
                commit=commit,
            )

        storage_scope = _legacy_scope_to_canonical(scope)
        timestamp = self._timestamp()
        await self._mark_retrieval_surfaces_stale_for_memory(
            user_id=user_id,
            memory_id=mirror_id,
            timestamp=timestamp,
        )
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET workspace_id = ?,
                conversation_id = ?,
                assistant_mode_id = ?,
                user_persona_id = ?,
                platform_id = ?,
                character_id = ?,
                scope = ?,
                scope_canonical = ?,
                canonical_text = ?,
                index_text = ?,
                payload_json = ?,
                source_kind = ?,
                confidence = ?,
                stability = ?,
                vitality = ?,
                maya_score = ?,
                privacy_level = ?,
                memory_category = ?,
                intimacy_boundary = ?,
                intimacy_boundary_confidence = ?,
                sensitivity = ?,
                themes_json = ?,
                platform_locked = ?,
                platform_id_lock = ?,
                preserve_verbatim = ?,
                language_codes_json = ?,
                status = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                workspace_id,
                conversation_id,
                assistant_mode_id,
                user_persona_id,
                platform_id,
                character_id,
                storage_scope,
                scope_canonical or storage_scope,
                summary_text,
                index_text,
                _encode_json(normalized_payload),
                MemorySourceKind.SUMMARIZED.value,
                confidence,
                stability,
                vitality,
                maya_score,
                privacy_level,
                MemoryCategory.UNKNOWN.value,
                intimacy_boundary.value,
                float(intimacy_boundary_confidence),
                (
                    sensitivity
                    or _derive_sensitivity_from_privacy(
                        privacy_level,
                        intimacy_boundary,
                        MemoryCategory.UNKNOWN,
                    )
                ).value,
                _encode_json(list(themes or [])),
                int(platform_locked),
                platform_id_lock,
                0,
                language_codes_json,
                status.value,
                timestamp,
                mirror_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        refreshed = await self.get_memory_object(mirror_id, user_id)
        if refreshed is None:
            raise ValueError(f"Unknown summary mirror memory_id: {mirror_id}")
        return refreshed

    async def latest_summary_mirror_payload(
        self,
        *,
        user_id: str,
        summary_kind: SummaryViewKind,
    ) -> dict[str, Any] | None:
        row = await self._fetch_one(
            """
            SELECT payload_json
            FROM memory_objects
            WHERE user_id = ?
              AND json_extract(payload_json, '$.summary_kind') = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id, summary_kind.value),
        )
        if row is None:
            return None
        payload = row.get("payload_json")
        return payload if isinstance(payload, dict) else None

    async def create_memory_object(
        self,
        *,
        user_id: str,
        object_type: MemoryObjectType,
        scope: MemoryScope,
        canonical_text: str,
        index_text: str | None = None,
        source_kind: MemorySourceKind,
        confidence: float,
        privacy_level: int,
        payload: dict[str, Any] | None = None,
        extraction_hash: str | None = None,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        stability: float = 0.5,
        vitality: float = 0.0,
        maya_score: float = 0.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        memory_category: MemoryCategory = MemoryCategory.UNKNOWN,
        intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        preserve_verbatim: bool = False,
        valid_from: str | None = None,
        valid_to: str | None = None,
        temporal_type: str = "unknown",
        language_codes: list[str] | None = None,
        memory_id: str | None = None,
        commit: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        auto_expires: bool | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        scope_canonical: str | None = None,
        active_presence_id: str | None = None,
        source_presence_id: str | None = None,
        presence_cluster_id: str | None = None,
        space_id: str | None = None,
        space_boundary_mode: str | None = None,
        memory_owner_id: str | None = None,
        source_mind_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
    ) -> dict[str, Any]:
        created, _was_created = await self._create_memory_object_impl(
            user_id=user_id,
            object_type=object_type,
            scope=scope,
            canonical_text=canonical_text,
            index_text=index_text,
            source_kind=source_kind,
            confidence=confidence,
            privacy_level=privacy_level,
            payload=payload,
            extraction_hash=extraction_hash,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
            stability=stability,
            vitality=vitality,
            maya_score=maya_score,
            status=status,
            memory_category=memory_category,
            intimacy_boundary=intimacy_boundary,
            intimacy_boundary_confidence=intimacy_boundary_confidence,
            preserve_verbatim=preserve_verbatim,
            valid_from=valid_from,
            valid_to=valid_to,
            temporal_type=temporal_type,
            language_codes=language_codes,
            memory_id=memory_id,
            commit=commit,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            sensitivity=sensitivity,
            themes=themes,
            auto_expires=auto_expires,
            platform_locked=platform_locked,
            platform_id_lock=platform_id_lock,
            scope_canonical=scope_canonical,
            active_presence_id=active_presence_id,
            source_presence_id=source_presence_id,
            presence_cluster_id=presence_cluster_id,
            space_id=space_id,
            space_boundary_mode=space_boundary_mode,
            memory_owner_id=memory_owner_id,
            source_mind_id=source_mind_id,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
        )
        return created

    async def create_memory_object_with_flag(
        self,
        *,
        user_id: str,
        object_type: MemoryObjectType,
        scope: MemoryScope,
        canonical_text: str,
        index_text: str | None = None,
        source_kind: MemorySourceKind,
        confidence: float,
        privacy_level: int,
        payload: dict[str, Any] | None = None,
        extraction_hash: str | None = None,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        stability: float = 0.5,
        vitality: float = 0.0,
        maya_score: float = 0.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        memory_category: MemoryCategory = MemoryCategory.UNKNOWN,
        intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        preserve_verbatim: bool = False,
        valid_from: str | None = None,
        valid_to: str | None = None,
        temporal_type: str = "unknown",
        language_codes: list[str] | None = None,
        memory_id: str | None = None,
        commit: bool = True,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        auto_expires: bool | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        scope_canonical: str | None = None,
        active_presence_id: str | None = None,
        source_presence_id: str | None = None,
        presence_cluster_id: str | None = None,
        space_id: str | None = None,
        space_boundary_mode: str | None = None,
        memory_owner_id: str | None = None,
        source_mind_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        return await self._create_memory_object_impl(
            user_id=user_id,
            object_type=object_type,
            scope=scope,
            canonical_text=canonical_text,
            index_text=index_text,
            source_kind=source_kind,
            confidence=confidence,
            privacy_level=privacy_level,
            payload=payload,
            extraction_hash=extraction_hash,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            assistant_mode_id=assistant_mode_id,
            stability=stability,
            vitality=vitality,
            maya_score=maya_score,
            status=status,
            memory_category=memory_category,
            intimacy_boundary=intimacy_boundary,
            intimacy_boundary_confidence=intimacy_boundary_confidence,
            preserve_verbatim=preserve_verbatim,
            valid_from=valid_from,
            valid_to=valid_to,
            temporal_type=temporal_type,
            language_codes=language_codes,
            memory_id=memory_id,
            commit=commit,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            sensitivity=sensitivity,
            themes=themes,
            auto_expires=auto_expires,
            platform_locked=platform_locked,
            platform_id_lock=platform_id_lock,
            scope_canonical=scope_canonical,
            active_presence_id=active_presence_id,
            source_presence_id=source_presence_id,
            presence_cluster_id=presence_cluster_id,
            space_id=space_id,
            space_boundary_mode=space_boundary_mode,
            memory_owner_id=memory_owner_id,
            source_mind_id=source_mind_id,
            embodiment_id=embodiment_id,
            realm_id=realm_id,
        )

    async def _create_memory_object_impl(
        self,
        *,
        user_id: str,
        object_type: MemoryObjectType,
        scope: MemoryScope,
        canonical_text: str,
        index_text: str | None = None,
        source_kind: MemorySourceKind,
        confidence: float,
        privacy_level: int,
        payload: dict[str, Any] | None = None,
        extraction_hash: str | None = None,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        assistant_mode_id: str | None = None,
        stability: float = 0.5,
        vitality: float = 0.0,
        maya_score: float = 0.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        memory_category: MemoryCategory = MemoryCategory.UNKNOWN,
        intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY,
        intimacy_boundary_confidence: float = 0.0,
        preserve_verbatim: bool = False,
        valid_from: str | None = None,
        valid_to: str | None = None,
        temporal_type: str = "unknown",
        language_codes: list[str] | None = None,
        memory_id: str | None = None,
        commit: bool = True,
        # Namespace redesign fields. Callers that don't supply them get
        # sensible defaults derived from privacy_level / scope so legacy
        # writers populate the new columns without changes.
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        sensitivity: MemorySensitivity | None = None,
        themes: list[str] | None = None,
        auto_expires: bool | None = None,
        platform_locked: bool = False,
        platform_id_lock: str | None = None,
        scope_canonical: str | None = None,
        active_presence_id: str | None = None,
        source_presence_id: str | None = None,
        presence_cluster_id: str | None = None,
        space_id: str | None = None,
        space_boundary_mode: str | None = None,
        memory_owner_id: str | None = None,
        source_mind_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        resolved_memory_id = memory_id or new_memory_id()
        timestamp = self._timestamp()
        language_codes_json = _encode_language_codes(language_codes)

        resolved_sensitivity = (
            sensitivity if sensitivity is not None
            else _derive_sensitivity_from_privacy(privacy_level, intimacy_boundary, memory_category)
        )
        resolved_themes_json = _encode_json(list(themes or []))
        resolved_auto_expires = (
            int(auto_expires) if auto_expires is not None
            else (1 if scope is MemoryScope.EPHEMERAL_SESSION else 0)
        )
        resolved_storage_scope = _legacy_scope_to_canonical(scope)
        resolved_scope_canonical = scope_canonical or resolved_storage_scope
        if resolved_scope_canonical not in {"chat", "character", "user"}:
            resolved_scope_canonical = resolved_storage_scope

        parameters = (
            resolved_memory_id,
            user_id,
            workspace_id,
            conversation_id,
            assistant_mode_id,
            object_type.value,
            resolved_storage_scope,
            canonical_text,
            index_text,
            extraction_hash,
            _encode_json(payload),
            source_kind.value,
            confidence,
            stability,
            vitality,
            maya_score,
            privacy_level,
            memory_category.value,
            intimacy_boundary.value,
            float(intimacy_boundary_confidence),
            int(preserve_verbatim),
            valid_from,
            valid_to,
            temporal_type,
            language_codes_json,
            status.value,
            timestamp,
            timestamp,
            user_persona_id,
            platform_id,
            character_id,
            resolved_sensitivity.value,
            resolved_themes_json,
            resolved_auto_expires,
            int(platform_locked),
            platform_id_lock,
            resolved_scope_canonical,
            active_presence_id,
            source_presence_id,
            presence_cluster_id,
            space_id,
            space_boundary_mode,
            memory_owner_id,
            source_mind_id,
            embodiment_id,
            realm_id,
        )
        try:
            await self._connection.execute(
                """
                INSERT INTO memory_objects(
                    id,
                    user_id,
                    workspace_id,
                    conversation_id,
                    assistant_mode_id,
                    object_type,
                    scope,
                    canonical_text,
                    index_text,
                    extraction_hash,
                    payload_json,
                    source_kind,
                    confidence,
                    stability,
                    vitality,
                    maya_score,
                    privacy_level,
                    memory_category,
                    intimacy_boundary,
                    intimacy_boundary_confidence,
                    preserve_verbatim,
                    valid_from,
                    valid_to,
                    temporal_type,
                    language_codes_json,
                    status,
                    created_at,
                    updated_at,
                    user_persona_id,
                    platform_id,
                    character_id,
                    sensitivity,
                    themes_json,
                    auto_expires,
                    platform_locked,
                    platform_id_lock,
                    scope_canonical,
                    active_presence_id,
                    source_presence_id,
                    presence_cluster_id,
                    space_id,
                    space_boundary_mode,
                    memory_owner_id,
                    source_mind_id,
                    embodiment_id,
                    realm_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                parameters,
            )
        except aiosqlite.IntegrityError:
            if extraction_hash is None:
                raise
            existing = await self.get_memory_object_by_extraction_hash(user_id, extraction_hash)
            if existing is None:
                raise
            return existing, False
        if commit:
            await self._connection.commit()
        created = await self._fetch_one(
            "SELECT * FROM memory_objects WHERE id = ?",
            (resolved_memory_id,),
        )
        if created is None:
            raise RuntimeError(f"Failed to create memory object {resolved_memory_id}")
        return created, True

    async def get_memory_object_by_extraction_hash(
        self,
        user_id: str,
        extraction_hash: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND extraction_hash = ?
            """,
            (user_id, extraction_hash),
        )

    async def filter_owned_presence_ids(
        self,
        *,
        user_id: str,
        presence_ids: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for presence_id in presence_ids:
            value = str(presence_id).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        if not normalized:
            return []
        placeholders = ", ".join("?" for _ in normalized)
        rows = await self._fetch_all(
            f"""
            SELECT id
            FROM presences
            WHERE owner_user_id = ?
              AND id IN ({placeholders})
            ORDER BY id ASC
            """,
            (user_id, *normalized),
        )
        allowed = {str(row["id"]) for row in rows}
        return [presence_id for presence_id in normalized if presence_id in allowed]

    async def add_memory_object_subjects(
        self,
        *,
        user_id: str,
        memory_id: str,
        subject_presence_ids: list[str],
        relation: str = "subject",
        commit: bool = True,
    ) -> None:
        allowed_ids = await self.filter_owned_presence_ids(
            user_id=user_id,
            presence_ids=subject_presence_ids,
        )
        if not allowed_ids:
            return
        timestamp = self._timestamp()
        for subject_presence_id in allowed_ids:
            await self._connection.execute(
                """
                INSERT OR IGNORE INTO memory_object_subjects(
                    memory_object_id,
                    owner_user_id,
                    subject_presence_id,
                    relation,
                    created_at
                )
                SELECT ?, ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1
                    FROM memory_objects
                    WHERE id = ?
                      AND user_id = ?
                )
                """,
                (
                    memory_id,
                    user_id,
                    subject_presence_id,
                    relation,
                    timestamp,
                    memory_id,
                    user_id,
                ),
            )
        if commit:
            await self._connection.commit()

    async def find_memory_object_for_extraction_merge(
        self,
        *,
        user_id: str,
        canonical_text: str,
        object_type: MemoryObjectType,
        scope: MemoryScope,
        user_persona_id: str | None,
        character_id: str | None,
        conversation_id: str | None,
        active_presence_id: str | None = None,
        source_presence_id: str | None = None,
        space_id: str | None = None,
        memory_owner_id: str | None = None,
        source_mind_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Find an existing same-namespace row when hashes differ by locks.

        Extraction hashes include platform-lock state after Phase 6. This
        lookup prevents a later unlocked restatement from creating a broad
        duplicate of an already locked memory.
        """

        return await self._fetch_one(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND object_type = ?
              AND COALESCE(scope_canonical, scope) = ?
              AND LOWER(TRIM(canonical_text)) = LOWER(TRIM(?))
              AND user_persona_id IS ?
              AND character_id IS ?
              AND conversation_id IS ?
              AND active_presence_id IS ?
              AND source_presence_id IS ?
              AND space_id IS ?
              AND memory_owner_id IS ?
              AND source_mind_id IS ?
              AND embodiment_id IS ?
              AND realm_id IS ?
            ORDER BY platform_locked DESC, updated_at DESC, _rowid DESC
            LIMIT 1
            """,
            (
                user_id,
                object_type.value,
                _legacy_scope_to_canonical(scope),
                canonical_text,
                user_persona_id,
                character_id,
                conversation_id,
                active_presence_id,
                source_presence_id,
                space_id,
                memory_owner_id,
                source_mind_id,
                embodiment_id,
                realm_id,
            ),
        )

    async def refresh_memory_object_provenance(
        self,
        *,
        user_id: str,
        memory_id: str,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
        source_message_ids: list[str],
        active_presence_id: str | None = None,
        source_presence_id: str | None = None,
        presence_cluster_id: str | None = None,
        space_id: str | None = None,
        space_boundary_mode: str | None = None,
        memory_owner_id: str | None = None,
        source_mind_id: str | None = None,
        embodiment_id: str | None = None,
        realm_id: str | None = None,
        touch: bool = True,
        commit: bool = True,
    ) -> dict[str, Any]:
        existing = await self.get_memory_object(memory_id, user_id)
        if existing is None:
            raise ValueError(f"Unknown memory_id: {memory_id}")

        payload = existing.get("payload_json") or {}
        normalized_payload = dict(payload) if isinstance(payload, dict) else {}
        current_source_ids = [
            str(item).strip()
            for item in normalized_payload.get("source_message_ids", [])
            if str(item).strip()
        ]
        merged_source_ids = _stable_string_union(current_source_ids, source_message_ids)
        source_ids_changed = merged_source_ids != current_source_ids
        normalized_payload["source_message_ids"] = merged_source_ids
        if source_ids_changed:
            normalized_payload["confirmation_count"] = int(normalized_payload.get("confirmation_count", 0)) + 1

        identifiers_changed = any(
            existing.get(key) != value
            for key, value in {
                "assistant_mode_id": assistant_mode_id,
                "workspace_id": workspace_id,
                "conversation_id": conversation_id,
                "active_presence_id": active_presence_id
                if existing.get("active_presence_id") is None
                else existing.get("active_presence_id"),
                "source_presence_id": source_presence_id
                if existing.get("source_presence_id") is None
                else existing.get("source_presence_id"),
                "presence_cluster_id": presence_cluster_id
                if existing.get("presence_cluster_id") is None
                else existing.get("presence_cluster_id"),
                "space_id": space_id
                if existing.get("space_id") is None
                else existing.get("space_id"),
                "space_boundary_mode": space_boundary_mode
                if existing.get("space_boundary_mode") is None
                else existing.get("space_boundary_mode"),
                "memory_owner_id": memory_owner_id
                if existing.get("memory_owner_id") is None
                else existing.get("memory_owner_id"),
                "source_mind_id": source_mind_id
                if existing.get("source_mind_id") is None
                else existing.get("source_mind_id"),
                "embodiment_id": embodiment_id
                if existing.get("embodiment_id") is None
                else existing.get("embodiment_id"),
                "realm_id": realm_id
                if existing.get("realm_id") is None
                else existing.get("realm_id"),
            }.items()
        )
        payload_changed = normalized_payload != payload
        if not (touch or identifiers_changed or payload_changed):
            return existing

        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET assistant_mode_id = ?,
                workspace_id = ?,
                conversation_id = ?,
                active_presence_id = COALESCE(active_presence_id, ?),
                source_presence_id = COALESCE(source_presence_id, ?),
                presence_cluster_id = COALESCE(presence_cluster_id, ?),
                space_id = COALESCE(space_id, ?),
                space_boundary_mode = COALESCE(space_boundary_mode, ?),
                memory_owner_id = COALESCE(memory_owner_id, ?),
                source_mind_id = COALESCE(source_mind_id, ?),
                embodiment_id = COALESCE(embodiment_id, ?),
                realm_id = COALESCE(realm_id, ?),
                payload_json = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                assistant_mode_id,
                workspace_id,
                conversation_id,
                active_presence_id,
                source_presence_id,
                presence_cluster_id,
                space_id,
                space_boundary_mode,
                memory_owner_id,
                source_mind_id,
                embodiment_id,
                realm_id,
                _encode_json(normalized_payload),
                timestamp,
                memory_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        refreshed = await self.get_memory_object(memory_id, user_id)
        if refreshed is None:
            raise ValueError(f"Unknown memory_id: {memory_id}")
        return refreshed

    async def merge_memory_object_write_restrictions(
        self,
        *,
        user_id: str,
        memory_id: str,
        privacy_level: int,
        intimacy_boundary: IntimacyBoundary,
        intimacy_boundary_confidence: float,
        sensitivity: MemorySensitivity,
        themes: list[str],
        auto_expires: bool,
        platform_locked: bool,
        platform_id_lock: str | None,
        extraction_hash: str | None = None,
        review_required: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Tighten an existing row after an extraction dedupe hit.

        This is deliberately one-way: later occurrences can make an existing
        memory more restricted, but cannot make it less sensitive, unlock it,
        remove expiry, or broaden gate themes.
        """

        existing = await self.get_memory_object(memory_id, user_id)
        if existing is None:
            raise ValueError(f"Unknown memory_id: {memory_id}")

        existing_sensitivity = MemorySensitivity(str(existing.get("sensitivity") or "unknown"))
        resolved_sensitivity = _max_sensitivity(existing_sensitivity, sensitivity)
        existing_boundary = IntimacyBoundary(str(existing.get("intimacy_boundary") or "ordinary"))
        resolved_boundary = _max_intimacy_boundary(existing_boundary, intimacy_boundary)
        existing_platform_locked = bool(int(existing.get("platform_locked") or 0))
        resolved_platform_locked = existing_platform_locked or platform_locked
        existing_lock = existing.get("platform_id_lock")
        resolved_platform_id_lock = existing_lock
        resolved_status = str(existing.get("status"))

        if platform_locked:
            if existing_platform_locked and existing_lock and platform_id_lock and existing_lock != platform_id_lock:
                resolved_status = MemoryStatus.REVIEW_REQUIRED.value
            elif not existing_platform_locked:
                resolved_platform_id_lock = platform_id_lock
            elif existing_lock is None:
                resolved_platform_id_lock = platform_id_lock
        if review_required and resolved_status == MemoryStatus.ACTIVE.value:
            resolved_status = MemoryStatus.REVIEW_REQUIRED.value

        existing_themes = existing.get("themes_json") or []
        if not isinstance(existing_themes, list):
            existing_themes = []
        if _SENSITIVITY_RANK[resolved_sensitivity] > _SENSITIVITY_RANK[existing_sensitivity]:
            resolved_themes = themes or [str(theme) for theme in existing_themes]
        elif existing_themes:
            resolved_themes = [str(theme) for theme in existing_themes]
        else:
            resolved_themes = themes

        resolved_privacy_level = max(int(existing.get("privacy_level") or 0), int(privacy_level))
        resolved_auto_expires = bool(int(existing.get("auto_expires") or 0)) or auto_expires
        resolved_extraction_hash = extraction_hash or existing.get("extraction_hash")

        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET privacy_level = ?,
                intimacy_boundary = ?,
                intimacy_boundary_confidence = ?,
                sensitivity = ?,
                themes_json = ?,
                auto_expires = ?,
                platform_locked = ?,
                platform_id_lock = ?,
                extraction_hash = ?,
                status = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                resolved_privacy_level,
                resolved_boundary.value,
                max(
                    float(existing.get("intimacy_boundary_confidence") or 0.0),
                    float(intimacy_boundary_confidence),
                ),
                resolved_sensitivity.value,
                _encode_json(resolved_themes),
                1 if resolved_auto_expires else 0,
                1 if resolved_platform_locked else 0,
                resolved_platform_id_lock,
                resolved_extraction_hash,
                resolved_status,
                timestamp,
                memory_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        refreshed = await self.get_memory_object(memory_id, user_id)
        if refreshed is None:
            raise ValueError(f"Unknown memory_id: {memory_id}")
        return refreshed

    async def fill_missing_memory_object_language_codes(
        self,
        *,
        user_id: str,
        memory_id: str,
        language_codes: list[str],
        commit: bool = True,
    ) -> dict[str, Any]:
        """Fill language metadata only when an existing row has none."""
        existing = await self.get_memory_object(memory_id, user_id)
        if existing is None:
            raise ValueError(f"Unknown memory_id: {memory_id}")
        encoded_language_codes = _encode_language_codes(language_codes)
        if encoded_language_codes is None:
            return existing

        timestamp = self._timestamp()
        await self._connection.execute(
            """
            UPDATE memory_objects
            SET language_codes_json = ?,
                updated_at = ?
            WHERE id = ?
              AND user_id = ?
              AND (
                    language_codes_json IS NULL
                    OR TRIM(language_codes_json) = ''
                    OR json_valid(language_codes_json) = 0
                    OR (
                        json_valid(language_codes_json) = 1
                        AND json_array_length(language_codes_json) = 0
                    )
                  )
            """,
            (
                encoded_language_codes,
                timestamp,
                memory_id,
                user_id,
            ),
        )
        if commit:
            await self._connection.commit()
        refreshed = await self.get_memory_object(memory_id, user_id)
        if refreshed is None:
            raise ValueError(f"Unknown memory_id: {memory_id}")
        return refreshed

    async def count_for_user_scopes(
        self,
        user_id: str,
        scopes: list[MemoryScope],
    ) -> int:
        return await self.count_for_context(
            user_id,
            scopes,
            workspace_id=None,
            conversation_id=None,
            assistant_mode_id=None,
        )

    async def count_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        sensitivity_gates_enabled: bool = False,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> int:
        if platform_id is not None and conversation_id is not None:
            clauses, parameters = self.namespace_visibility_clauses(
                scopes,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                sensitivity_gates_enabled=sensitivity_gates_enabled,
                table_alias="memory_objects",
            )
            clause_joiner = " AND "
        else:
            clauses, parameters = self._context_scope_clauses(
                scopes,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                assistant_mode_id=assistant_mode_id,
            )
            clause_joiner = " OR "
        if not clauses:
            return 0
        mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            alias="memory_objects",
        )
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
            active_embodiment_id=active_embodiment_id,
            alias="memory_objects",
        )
        realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
            active_realm_id=active_realm_id,
            alias="memory_objects",
        )

        cursor = await self._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM memory_objects
            WHERE user_id = ?
              AND (
                  {clauses}
              )
              AND status IN ({status_placeholders})
              AND archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND {mind_clause}
              AND {embodiment_clause}
              AND {realm_clause}
            """.format(
                clauses=clause_joiner.join(clauses),
                status_placeholders=", ".join("?" for _ in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
                visibility_clause=conversation_visibility_clause("memory_objects"),
                mind_clause=mind_clause,
                embodiment_clause=embodiment_clause,
                realm_clause=realm_clause,
            ),
            tuple(
                [
                    user_id,
                    *parameters,
                    *(status.value for status in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
                    conversation_id,
                    *mind_parameters,
                    *embodiment_parameters,
                    *realm_parameters,
                ]
            ),
        )
        row = await cursor.fetchone()
        return int(row["count"])

    async def sum_canonical_text_length_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
        privacy_ceiling: int,
        allow_intimacy_context: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        sensitivity_gates_enabled: bool = False,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> int:
        """Return the total canonical_text length for eligible active memories.

        The filter mirrors retrieval eligibility: user_id, allowed scopes,
        active status, and privacy ceiling. Used to detect small corpora
        where the full memory set fits inside the context budget.
        """
        if platform_id is not None and conversation_id is not None:
            clauses, parameters = self.namespace_visibility_clauses(
                scopes,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                sensitivity_gates_enabled=sensitivity_gates_enabled,
                table_alias="memory_objects",
            )
            clause_joiner = " AND "
        else:
            clauses, parameters = self._context_scope_clauses(
                scopes,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                assistant_mode_id=assistant_mode_id,
            )
            clause_joiner = " OR "
        if not clauses:
            return 0
        space_clause, space_parameters = space_visibility_sql_clause_for_context(
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            alias="memory_objects",
        )
        mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            alias="memory_objects",
        )
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
            active_embodiment_id=active_embodiment_id,
            alias="memory_objects",
        )
        realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
            active_realm_id=active_realm_id,
            alias="memory_objects",
        )

        cursor = await self._connection.execute(
            """
            SELECT COALESCE(SUM(LENGTH(canonical_text)), 0) AS total_length
            FROM memory_objects
            WHERE user_id = ?
              AND (
                  {clauses}
              )
              AND status IN ({status_placeholders})
              AND archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND {space_clause}
              AND {mind_clause}
              AND {embodiment_clause}
              AND {realm_clause}
              AND privacy_level <= ?
              AND {intimacy_filter}
            """.format(
                clauses=clause_joiner.join(clauses),
                status_placeholders=", ".join("?" for _ in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
                visibility_clause=conversation_visibility_clause("memory_objects"),
                space_clause=space_clause,
                mind_clause=mind_clause,
                embodiment_clause=embodiment_clause,
                realm_clause=realm_clause,
                intimacy_filter=memory_object_intimacy_sql_clause(
                    "memory_objects",
                    allow_intimacy_context=allow_intimacy_context,
                ),
            ),
            tuple(
                [
                    user_id,
                    *parameters,
                    *(status.value for status in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
                    conversation_id,
                    *space_parameters,
                    *mind_parameters,
                    *embodiment_parameters,
                    *realm_parameters,
                    privacy_ceiling,
                ]
            ),
        )
        row = await cursor.fetchone()
        return int(row["total_length"])

    async def list_eligible_for_context(
        self,
        user_id: str,
        scopes: list[MemoryScope],
        *,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
        privacy_ceiling: int,
        allow_intimacy_context: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        sensitivity_gates_enabled: bool = False,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return eligible active memory objects for small-corpus composition."""
        if platform_id is not None and conversation_id is not None:
            clauses, parameters = self.namespace_visibility_clauses(
                scopes,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                sensitivity_gates_enabled=sensitivity_gates_enabled,
                table_alias="memory_objects",
            )
            clause_joiner = " AND "
        else:
            clauses, parameters = self._context_scope_clauses(
                scopes,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                assistant_mode_id=assistant_mode_id,
            )
            clause_joiner = " OR "
        if not clauses:
            return []
        space_clause, space_parameters = space_visibility_sql_clause_for_context(
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            alias="memory_objects",
        )
        mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            alias="memory_objects",
        )
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
            active_embodiment_id=active_embodiment_id,
            alias="memory_objects",
        )
        realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
            active_realm_id=active_realm_id,
            alias="memory_objects",
        )

        rows = await self._fetch_all(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND (
                  {clauses}
              )
              AND status IN ({status_placeholders})
              AND archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND {space_clause}
              AND {mind_clause}
              AND {embodiment_clause}
              AND {realm_clause}
              AND privacy_level <= ?
              AND {intimacy_filter}
            ORDER BY updated_at DESC, id ASC
            """.format(
                clauses=clause_joiner.join(clauses),
                status_placeholders=", ".join("?" for _ in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
                visibility_clause=conversation_visibility_clause("memory_objects"),
                space_clause=space_clause,
                mind_clause=mind_clause,
                embodiment_clause=embodiment_clause,
                realm_clause=realm_clause,
                intimacy_filter=memory_object_intimacy_sql_clause(
                    "memory_objects",
                    allow_intimacy_context=allow_intimacy_context,
                ),
            ),
            tuple(
                [
                    user_id,
                    *parameters,
                    *(status.value for status in RETRIEVAL_ELIGIBLE_MEMORY_STATUSES),
                    conversation_id,
                    *space_parameters,
                    *mind_parameters,
                    *embodiment_parameters,
                    *realm_parameters,
                    privacy_ceiling,
                ]
            ),
        )
        await annotate_realm_bridge_modes_for_rows(
            self._connection,
            rows,
            active_realm_id=active_realm_id,
        )
        await annotate_overseer_grants_for_rows(
            self._connection,
            rows,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
        )
        return rows

    @staticmethod
    def namespace_scope_clauses(
        scopes: list[MemoryScope],
        *,
        user_persona_id: str | None,
        character_id: str | None,
        conversation_id: str,
        remember_across_chats: bool,
        incognito: bool,
        table_alias: str = "memory_objects",
        allow_cross_conversation_chat: bool = False,
    ) -> tuple[list[str], list[Any]]:
        """Build namespace-aware scope filters for the canonical scopes.

        Returns OR-able clauses that retrieval can combine with the user
        filter. Unlike the legacy ``_context_scope_clauses``, this helper
        works against ``scope_canonical`` (populated by Phase 1 backfill /
        Phase 3 writes) and enforces persona / character / conversation
        ownership directly.

        Cross-chat eligibility (character + user scopes) requires both
        ``incognito=False`` and ``remember_across_chats=True``; otherwise
        only the chat scope clause is emitted regardless of what the
        caller asked for.
        """

        prefix = f"{table_alias}." if table_alias else ""
        cross_chat = (not incognito) and remember_across_chats

        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in scopes:
            if scope is MemoryScope.CHAT:
                if allow_cross_conversation_chat and cross_chat:
                    clauses.append(
                        f"({prefix}scope_canonical = 'chat' "
                        f"AND ({prefix}user_persona_id IS ? "
                        f"OR ({prefix}user_persona_id IS NULL "
                        f"AND {prefix}conversation_id = ?)))"
                    )
                    parameters.extend([user_persona_id, conversation_id])
                else:
                    clauses.append(
                        f"({prefix}scope_canonical = 'chat' "
                        f"AND ({prefix}user_persona_id IS ? "
                        f"OR {prefix}user_persona_id IS NULL) "
                        f"AND {prefix}conversation_id = ?)"
                    )
                    parameters.extend([user_persona_id, conversation_id])
            elif scope is MemoryScope.CHARACTER and cross_chat and character_id is not None:
                clauses.append(
                    f"({prefix}scope_canonical = 'character' "
                    f"AND {prefix}user_persona_id IS ? "
                    f"AND {prefix}character_id = ?)"
                )
                parameters.extend([user_persona_id, character_id])
            elif scope is MemoryScope.USER and cross_chat:
                clauses.append(
                    f"({prefix}scope_canonical = 'user' "
                    f"AND {prefix}user_persona_id IS ?)"
                )
                parameters.append(user_persona_id)
        return clauses, parameters

    @staticmethod
    def canonical_retrieval_scopes(scopes: list[MemoryScope]) -> list[MemoryScope]:
        """Translate policy/storage scopes to the post-redesign reader scopes."""

        canonical: list[MemoryScope] = []
        for scope in scopes:
            if scope in {
                MemoryScope.CHAT,
                MemoryScope.CONVERSATION,
                MemoryScope.EPHEMERAL_SESSION,
            }:
                target = MemoryScope.CHAT
            elif scope in {MemoryScope.CHARACTER, MemoryScope.WORKSPACE}:
                target = MemoryScope.CHARACTER
            elif scope in {
                MemoryScope.USER,
                MemoryScope.GLOBAL_USER,
                MemoryScope.ASSISTANT_MODE,
            }:
                target = MemoryScope.USER
            else:
                continue
            if target not in canonical:
                canonical.append(target)
        return canonical

    @staticmethod
    def namespace_visibility_clauses(
        scopes: list[MemoryScope],
        *,
        user_persona_id: str | None,
        platform_id: str,
        character_id: str | None,
        conversation_id: str,
        remember_across_chats: bool,
        remember_across_devices: bool,
        incognito: bool,
        sensitivity_gates_enabled: bool = False,
        allow_private_sensitivity: bool = False,
        table_alias: str = "memory_objects",
        allow_cross_conversation_chat: bool = False,
    ) -> tuple[list[str], list[Any]]:
        """Build the Phase 7 namespace/platform/sensitivity reader filters."""

        scope_clauses, scope_parameters = MemoryObjectRepository.namespace_scope_clauses(
            MemoryObjectRepository.canonical_retrieval_scopes(scopes),
            user_persona_id=user_persona_id,
            character_id=character_id,
            conversation_id=conversation_id,
            remember_across_chats=remember_across_chats,
            incognito=incognito,
            table_alias=table_alias,
            allow_cross_conversation_chat=allow_cross_conversation_chat,
        )
        if not scope_clauses:
            return [], []
        platform_clause, platform_parameters = MemoryObjectRepository.platform_lock_clause(
            platform_id=platform_id,
            remember_across_devices=remember_across_devices,
            table_alias=table_alias,
        )
        return (
            [
                "(" + " OR ".join(scope_clauses) + ")",
                MemoryObjectRepository.sensitivity_filter_clause(
                    gates_enabled=sensitivity_gates_enabled,
                    allow_private_sensitivity=allow_private_sensitivity,
                    table_alias=table_alias,
                ),
                platform_clause,
            ],
            [*scope_parameters, *platform_parameters],
        )

    @staticmethod
    def sensitivity_filter_clause(
        *,
        gates_enabled: bool,
        allow_private_sensitivity: bool = False,
        table_alias: str = "memory_objects",
    ) -> str:
        """SQL fragment that gates retrieval by sensitivity.

        With ``gates_enabled=False`` (the v1 default chosen for Atagia) the
        clause hides ``unknown`` / ``private`` / ``secret`` rows
        fail-closed. They remain reachable through admin/review paths and
        through direct id lookup, but never through ordinary retrieval.

        With ``gates_enabled=True`` only ``unknown`` is hidden by default;
        ``private`` and ``secret`` rows become candidates that the
        retrieval pipeline must run through the LLM theme gate and the
        explicit-reference gate respectively.

        ``allow_private_sensitivity`` is narrower: it admits private rows
        while keeping secret rows excluded at SQL time.
        """

        prefix = f"{table_alias}." if table_alias else ""
        if allow_private_sensitivity:
            return f"({prefix}sensitivity IN ('public', 'private'))"
        if gates_enabled:
            return f"({prefix}sensitivity IN ('public', 'private', 'secret'))"
        return f"({prefix}sensitivity = 'public')"

    @staticmethod
    def platform_lock_clause(
        *,
        platform_id: str,
        remember_across_devices: bool,
        table_alias: str = "memory_objects",
    ) -> tuple[str, list[Any]]:
        """SQL fragment that enforces ``platform_locked`` and cross-device.

        Locked rows are visible only on their locked platform. When the
        user disabled cross-device memory, only rows that originated on or
        are locked to the active platform are eligible.
        """

        prefix = f"{table_alias}." if table_alias else ""
        if remember_across_devices:
            return (
                f"({prefix}platform_locked = 0 OR {prefix}platform_id_lock = ?)",
                [platform_id],
            )
        return (
            f"({prefix}platform_id_lock = ? "
            f"OR ({prefix}platform_locked = 0 AND {prefix}platform_id = ?))",
            [platform_id, platform_id],
        )

    @staticmethod
    def _context_scope_clauses(
        scopes: list[MemoryScope],
        *,
        workspace_id: str | None,
        conversation_id: str | None,
        assistant_mode_id: str | None,
    ) -> tuple[list[str], list[Any]]:
        del assistant_mode_id
        scope_expr = (
            "CASE "
            "WHEN scope_canonical IS NOT NULL THEN scope_canonical "
            "WHEN scope IN ('conversation', 'ephemeral_session') THEN 'chat' "
            "WHEN scope = 'workspace' THEN 'character' "
            "WHEN scope IN ('global_user', 'assistant_mode') THEN 'user' "
            "ELSE scope END"
        )
        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in MemoryObjectRepository.canonical_retrieval_scopes(scopes):
            if scope is MemoryScope.USER:
                clauses.append(f"({scope_expr} = 'user')")
            elif scope is MemoryScope.CHARACTER and workspace_id is not None:
                clauses.append(
                    f"({scope_expr} = 'character' "
                    "AND (character_id = ? OR (character_id IS NULL AND workspace_id = ?)))"
                )
                parameters.extend([workspace_id, workspace_id])
            elif scope is MemoryScope.CHAT and conversation_id is not None:
                clauses.append(f"({scope_expr} = 'chat' AND conversation_id = ?)")
                parameters.append(conversation_id)
        return clauses, parameters

    async def get_state_snapshot(
        self,
        user_id: str,
        *,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
        allow_intimacy_context: bool = False,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        sensitivity_gates_enabled: bool = False,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
    ) -> dict[str, Any]:
        if platform_id is not None and conversation_id is not None:
            clauses, parameters = self.namespace_visibility_clauses(
                [
                    MemoryScope.CHAT,
                    MemoryScope.CHARACTER,
                    MemoryScope.USER,
                ],
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id,
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                sensitivity_gates_enabled=sensitivity_gates_enabled,
                table_alias="memory_objects",
            )
            clause_joiner = " AND "
        else:
            clauses, parameters = self._state_context_clauses(
                assistant_mode_id=assistant_mode_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
            )
            clause_joiner = " OR "
        if not clauses:
            return {}
        space_clause, space_parameters = space_visibility_sql_clause_for_context(
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            alias="memory_objects",
        )
        mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            alias="memory_objects",
            allow_overseer_grants=False,
        )
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
            active_embodiment_id=active_embodiment_id,
            alias="memory_objects",
        )
        realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
            active_realm_id=active_realm_id,
            alias="memory_objects",
            allowed_bridge_modes=APPLICABLE_REALM_BRIDGE_MODES,
        )

        rows = await self._fetch_all(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND object_type = ?
              AND status = ?
              AND ({clauses})
              AND {space_clause}
              AND {mind_clause}
              AND {embodiment_clause}
              AND {realm_clause}
              AND archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND {intimacy_filter}
            ORDER BY updated_at DESC, id ASC
            """.format(
                clauses=clause_joiner.join(clauses),
                space_clause=space_clause,
                mind_clause=mind_clause,
                embodiment_clause=embodiment_clause,
                realm_clause=realm_clause,
                visibility_clause=conversation_visibility_clause("memory_objects"),
                intimacy_filter=memory_object_intimacy_sql_clause(
                    "memory_objects",
                    allow_intimacy_context=allow_intimacy_context,
                ),
            ),
            (
                user_id,
                MemoryObjectType.STATE_SNAPSHOT.value,
                MemoryStatus.ACTIVE.value,
                *parameters,
                *space_parameters,
                *mind_parameters,
                *embodiment_parameters,
                *realm_parameters,
                conversation_id,
            ),
        )

        resolved: dict[str, tuple[int, str, Any]] = {}
        now = self._clock.now()
        for row in rows:
            if self._is_expired_ephemeral_state_snapshot(row, now):
                continue
            scope_rank = self._state_scope_rank(
                str(row.get("scope_canonical") or row.get("scope") or "")
            )
            updated_at = str(row["updated_at"])
            payload = row.get("payload_json") or {}
            if not isinstance(payload, dict):
                continue
            row_realm_id = (
                str(row["realm_id"])
                if row.get("realm_id") is not None
                else None
            )
            for key, value in payload.items():
                current = resolved.get(key)
                resolved_value = value
                if (
                    active_realm_id is not None
                    and row_realm_id is not None
                    and row_realm_id != active_realm_id
                ):
                    resolved_value = {
                        "value": value,
                        "realm": {
                            "active_realm_id": row_realm_id,
                            "active_request_realm_id": active_realm_id,
                            "cross_realm_mode": "applicable",
                        },
                    }
                if current is None or (scope_rank, updated_at) > (current[0], current[1]):
                    resolved[key] = (scope_rank, updated_at, resolved_value)
        return {key: value for key, (_, _, value) in resolved.items()}

    async def has_memory_for_source_message(
        self,
        *,
        user_id: str,
        object_type: MemoryObjectType,
        source_message_id: str,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
        statuses: tuple[MemoryStatus, ...] | None = RETRIEVAL_ELIGIBLE_MEMORY_STATUSES,
    ) -> bool:
        clauses = [
            "mo.user_id = ?",
            "mo.object_type = ?",
            "source_ids.value = ?",
        ]
        parameters: list[Any] = [
            user_id,
            object_type.value,
            source_message_id,
        ]
        status_clause, status_parameters = _status_filter_clause("mo.status", statuses)
        if status_clause:
            clauses.append(status_clause)
            parameters.extend(status_parameters)
        if assistant_mode_id is not None:
            clauses.append("mo.assistant_mode_id = ?")
            parameters.append(assistant_mode_id)
        if workspace_id is not None:
            clauses.append("mo.workspace_id = ?")
            parameters.append(workspace_id)
        if conversation_id is not None:
            clauses.append("mo.conversation_id = ?")
            parameters.append(conversation_id)
        clauses.append("mo.archived_by_conversation_id IS NULL")
        clauses.append(conversation_visibility_clause("mo"))
        parameters.append(conversation_id)
        cursor = await self._connection.execute(
            """
            SELECT 1
            FROM memory_objects AS mo
            JOIN json_each(mo.payload_json, '$.source_message_ids') AS source_ids
              ON 1 = 1
            WHERE {clauses}
            LIMIT 1
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )
        row = await cursor.fetchone()
        return row is not None

    async def list_for_user(
        self,
        user_id: str,
        *,
        statuses: tuple[MemoryStatus, ...] | None = RETRIEVAL_ELIGIBLE_MEMORY_STATUSES,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        status_clause, status_parameters = _status_filter_clause("status", statuses)
        if status_clause:
            clauses.append(status_clause)
            parameters.extend(status_parameters)
        clauses.append("archived_by_conversation_id IS NULL")
        clauses.append(conversation_visibility_clause("memory_objects"))
        parameters.append(None)
        return await self._fetch_all(
            """
            SELECT *
            FROM memory_objects
            WHERE {clauses}
            ORDER BY created_at ASC, id ASC
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def list_for_source_message(
        self,
        *,
        user_id: str,
        source_message_id: str,
        assistant_mode_id: str | None = None,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        statuses: tuple[MemoryStatus, ...] | None = RETRIEVAL_ELIGIBLE_MEMORY_STATUSES,
    ) -> list[dict[str, Any]]:
        clauses = [
            "mo.user_id = ?",
            "source_ids.value = ?",
        ]
        parameters: list[Any] = [user_id, source_message_id]
        status_clause, status_parameters = _status_filter_clause("mo.status", statuses)
        if status_clause:
            clauses.append(status_clause)
            parameters.extend(status_parameters)
        if assistant_mode_id is not None:
            clauses.append("mo.assistant_mode_id = ?")
            parameters.append(assistant_mode_id)
        if workspace_id is not None:
            clauses.append("mo.workspace_id = ?")
            parameters.append(workspace_id)
        if conversation_id is not None:
            clauses.append("mo.conversation_id = ?")
            parameters.append(conversation_id)
        clauses.append("mo.archived_by_conversation_id IS NULL")
        clauses.append(conversation_visibility_clause("mo"))
        parameters.append(conversation_id)
        return await self._fetch_all(
            """
            SELECT DISTINCT mo.*
            FROM memory_objects AS mo
            JOIN json_each(mo.payload_json, '$.source_message_ids') AS source_ids
              ON 1 = 1
            WHERE {clauses}
            ORDER BY mo.created_at ASC, mo.id ASC
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def search_memory_objects(
        self,
        user_id: str,
        query: str,
        limit: int,
        *,
        statuses: tuple[MemoryStatus, ...] | None = RETRIEVAL_ELIGIBLE_MEMORY_STATUSES,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["mo.user_id = ?", "memory_objects_fts MATCH ?"]
        parameters: list[Any] = [user_id, query]
        status_clause, status_parameters = _status_filter_clause("mo.status", statuses)
        if status_clause:
            clauses.append(status_clause)
            parameters.extend(status_parameters)
        clauses.append("mo.archived_by_conversation_id IS NULL")
        clauses.append(conversation_visibility_clause("mo"))
        parameters.append(conversation_id)
        parameters.append(limit)
        return await self._fetch_all(
            """
            SELECT
                mo.*,
                bm25(memory_objects_fts) AS rank
            FROM memory_objects_fts
            JOIN memory_objects AS mo ON mo._rowid = memory_objects_fts.rowid
            WHERE {clauses}
            ORDER BY rank ASC, mo.created_at DESC
            LIMIT ?
            """.format(clauses=" AND ".join(clauses)),
            tuple(parameters),
        )

    @staticmethod
    def _state_context_clauses(
        *,
        assistant_mode_id: str | None,
        workspace_id: str | None,
        conversation_id: str | None,
    ) -> tuple[list[str], list[Any]]:
        del assistant_mode_id
        scope_expr = (
            "CASE "
            "WHEN scope_canonical IS NOT NULL THEN scope_canonical "
            "WHEN scope IN ('conversation', 'ephemeral_session') THEN 'chat' "
            "WHEN scope = 'workspace' THEN 'character' "
            "WHEN scope IN ('global_user', 'assistant_mode') THEN 'user' "
            "ELSE scope END"
        )
        clauses = [f"({scope_expr} = 'user')"]
        parameters: list[Any] = []
        if workspace_id is not None:
            clauses.append(
                f"({scope_expr} = 'character' "
                "AND (character_id = ? OR (character_id IS NULL AND workspace_id = ?)))"
            )
            parameters.extend([workspace_id, workspace_id])
        if conversation_id is not None:
            clauses.append(f"({scope_expr} = 'chat' AND conversation_id = ?)")
            parameters.append(conversation_id)
        return clauses, parameters

    @staticmethod
    def _state_scope_rank(scope_value: str) -> int:
        order = {
            MemoryScope.GLOBAL_USER.value: 0,
            MemoryScope.USER.value: 0,
            MemoryScope.WORKSPACE.value: 1,
            MemoryScope.CHARACTER.value: 1,
            MemoryScope.CONVERSATION.value: 2,
            MemoryScope.EPHEMERAL_SESSION.value: 3,
            MemoryScope.CHAT.value: 3,
        }
        return order.get(scope_value, -1)

    def _is_expired_ephemeral_state_snapshot(
        self,
        row: dict[str, Any],
        reference: datetime,
    ) -> bool:
        if str(row.get("temporal_type", "unknown")) != "ephemeral":
            return False
        valid_from = self._parse_temporal_datetime(row.get("valid_from"), reference)
        if valid_from is None:
            return True
        valid_to = self._parse_temporal_datetime(row.get("valid_to"), reference)
        effective_end = valid_to or (
            valid_from + timedelta(hours=self._settings.ephemeral_scoring_hours)
        )
        return effective_end < reference

    @staticmethod
    def _parse_temporal_datetime(value: Any, reference: datetime) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None and reference.tzinfo is not None:
            parsed = parsed.replace(tzinfo=reference.tzinfo)
        return parsed
