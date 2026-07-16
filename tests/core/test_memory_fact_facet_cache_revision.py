"""Cache-revision coverage for the fact/facet retrieval surface."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.memory_evidence_repository import MemoryEvidenceRepository
from atagia.core.memory_fact_facet_repository import MemoryFactFacetRepository
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
)
from atagia.models.schemas_memory import (
    MemoryEvidenceSupportKind,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
USER_ID = "usr_fact_cache_revision"
CONVERSATION_ID = "cnv_fact_cache_revision"
MEMORY_ID = "mem_fact_cache_revision"
MESSAGE_ID = "msg_fact_cache_revision"
SENTINEL_UPDATED_AT = "2000-01-01T00:00:00+00:00"


async def _seed_fact_sources(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
) -> str:
    await UserRepository(connection, clock).create_user(USER_ID)
    await connection.execute(
        """
        INSERT INTO assistant_modes(
            id,
            display_name,
            prompt_hash,
            memory_policy_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, '{}', ?, ?)
        """,
        (
            "coding_debug",
            "Coding Debug",
            "fact-cache-revision-hash",
            clock.now().isoformat(),
            clock.now().isoformat(),
        ),
    )
    await connection.commit()
    await ConversationRepository(connection, clock).create_conversation(
        CONVERSATION_ID,
        USER_ID,
        None,
        "coding_debug",
        "Fact cache revision",
    )
    await MessageRepository(connection, clock).create_message(
        MESSAGE_ID,
        CONVERSATION_ID,
        "user",
        1,
        "My current city is Paris.",
    )
    await MemoryObjectRepository(connection, clock).create_memory_object(
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text="The user's current city is Paris.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        memory_id=MEMORY_ID,
        payload={"source_message_ids": [MESSAGE_ID]},
    )
    packet = await MemoryEvidenceRepository(
        connection,
        clock,
    ).create_support_edge_with_spans(
        user_id=USER_ID,
        memory_id=MEMORY_ID,
        support_kind=MemoryEvidenceSupportKind.DIRECT,
        confidence=0.9,
        spans=[
            {
                "span_role": "source",
                "message_id": MESSAGE_ID,
                "conversation_id": CONVERSATION_ID,
                "quote_text": "My current city is Paris.",
            }
        ],
    )
    return str(packet["spans"][0]["id"])


async def _revision_state(connection: aiosqlite.Connection) -> dict[str, Any]:
    cursor = await connection.execute(
        """
        SELECT
            user_lifecycles.cache_revision,
            user_lifecycles.derivation_revision,
            user_lifecycles.source_revision,
            user_lifecycles.updated_at,
            conversation_lifecycles.source_revision AS conversation_source_revision
        FROM user_lifecycles
        JOIN conversation_lifecycles
          ON conversation_lifecycles.user_id = user_lifecycles.user_id
        WHERE user_lifecycles.user_id = ?
          AND conversation_lifecycles.conversation_id = ?
        """,
        (USER_ID, CONVERSATION_ID),
    )
    row = await cursor.fetchone()
    assert row is not None
    return dict(row)


async def _prepare_revision_assertion(
    connection: aiosqlite.Connection,
) -> dict[str, Any]:
    await connection.execute(
        """
        UPDATE user_lifecycles
        SET updated_at = ?
        WHERE user_id = ?
        """,
        (SENTINEL_UPDATED_AT, USER_ID),
    )
    await connection.commit()
    state = await _revision_state(connection)
    assert state["updated_at"] == SENTINEL_UPDATED_AT
    return state


async def _assert_cache_only_bump(
    connection: aiosqlite.Connection,
    before: dict[str, Any],
) -> None:
    after = await _revision_state(connection)
    assert after["cache_revision"] == before["cache_revision"] + 1
    assert after["updated_at"] != SENTINEL_UPDATED_AT
    assert after["derivation_revision"] == before["derivation_revision"]
    assert after["source_revision"] == before["source_revision"]
    assert (
        after["conversation_source_revision"]
        == (before["conversation_source_revision"])
    )


async def _insert_raw_fact_facet(
    connection: aiosqlite.Connection,
    *,
    source_span_id: str,
    fact_id: str,
) -> None:
    await connection.execute(
        """
        INSERT INTO memory_fact_facets(
            id,
            user_id,
            conversation_id,
            memory_id,
            source_message_id,
            source_span_id,
            source_hash,
            subject_surface,
            surface_class,
            facet_label,
            value_text,
            value_norm_key,
            assertion_kind,
            support_kind,
            observed_at,
            confidence,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fact_id,
            USER_ID,
            CONVERSATION_ID,
            MEMORY_ID,
            MESSAGE_ID,
            source_span_id,
            "raw-source-hash",
            "user",
            "structured",
            "location.current_city",
            "Paris",
            "paris",
            "evidence",
            "direct",
            "2026-07-13T10:00:00+00:00",
            0.9,
            "2026-07-13T10:00:00+00:00",
            "2026-07-13T10:00:00+00:00",
        ),
    )
    await connection.commit()


@pytest.mark.asyncio
async def test_repository_insert_and_update_advance_only_cache_revision() -> None:
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        source_span_id = await _seed_fact_sources(connection, clock)
        repository = MemoryFactFacetRepository(connection, clock)

        before_insert = await _prepare_revision_assertion(connection)
        inserted = await repository.upsert_fact_facet(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            memory_id=MEMORY_ID,
            source_span_id=source_span_id,
            source_message_id=MESSAGE_ID,
            subject_surface="user",
            surface_class="structured",
            facet_label="location.current_city",
            value_text="Paris",
            value_norm_key="paris",
            support_kind="direct",
            observed_at="2026-07-13T10:00:00+00:00",
            confidence=0.9,
        )
        await _assert_cache_only_bump(connection, before_insert)

        before_update = await _prepare_revision_assertion(connection)
        updated = await repository.upsert_fact_facet(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            memory_id=MEMORY_ID,
            source_span_id=source_span_id,
            source_message_id=MESSAGE_ID,
            subject_surface="user",
            surface_class="structured",
            facet_label="location.current_city",
            value_text="Paris, France",
            value_norm_key="paris",
            support_kind="direct",
            observed_at="2026-07-13T10:00:00+00:00",
            confidence=0.95,
        )
        await _assert_cache_only_bump(connection, before_update)
        assert updated["id"] == inserted["id"]
        assert updated["value_text"] == "Paris, France"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_raw_sql_insert_update_delete_advance_only_cache_revision() -> None:
    clock = FrozenClock(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc))
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        source_span_id = await _seed_fact_sources(connection, clock)
        fact_id = "mff_raw_cache_revision"

        before_insert = await _prepare_revision_assertion(connection)
        await _insert_raw_fact_facet(
            connection,
            source_span_id=source_span_id,
            fact_id=fact_id,
        )
        await _assert_cache_only_bump(connection, before_insert)

        before_update = await _prepare_revision_assertion(connection)
        await connection.execute(
            """
            UPDATE memory_fact_facets
            SET value_text = ?, confidence = ?, updated_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            (
                "Paris, France",
                0.95,
                "2026-07-13T10:01:00+00:00",
                fact_id,
                USER_ID,
            ),
        )
        await connection.commit()
        await _assert_cache_only_bump(connection, before_update)

        before_delete = await _prepare_revision_assertion(connection)
        await connection.execute(
            """
            DELETE FROM memory_fact_facets
            WHERE id = ?
              AND user_id = ?
            """,
            (fact_id, USER_ID),
        )
        await connection.commit()
        await _assert_cache_only_bump(connection, before_delete)
    finally:
        await connection.close()
