"""Tests for summary view persistence helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.core.summary_repository import SummaryRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import MemoryScope, SummaryViewKind

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 3, 10, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    summaries = SummaryRepository(connection, clock)
    await users.create_user("usr_1")
    await users.create_user("usr_2")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await workspaces.create_workspace("wrk_2", "usr_2", "Other Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", "coding_debug", "Chat")
    await conversations.create_conversation("cnv_2", "usr_2", "wrk_2", "coding_debug", "Other Chat")
    return connection, summaries


@pytest.mark.asyncio
async def test_create_summary_creates_row_with_correct_fields() -> None:
    connection, summaries = await _build_runtime()
    try:
        summary_id = await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_1",
                "conversation_id": "cnv_1",
                "workspace_id": "wrk_1",
                "source_message_start_seq": 1,
                "source_message_end_seq": 4,
                "summary_kind": "conversation_chunk",
                "summary_text": "Chunk summary",
                "source_object_ids_json": ["mem_1", "mem_2"],
                "maya_score": 1.5,
                "model": "classify-test-model",
                "created_at": "2026-04-03T10:00:00+00:00",
            }
        )

        row = await summaries.get_summary(summary_id, "usr_1")
        assert row is not None
        assert row["summary_text"] == "Chunk summary"
        assert row["source_object_ids_json"] == ["mem_1", "mem_2"]
        assert row["model"] == "classify-test-model"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_summary_rejects_missing_parent_references() -> None:
    connection, summaries = await _build_runtime()
    try:
        with pytest.raises(ValueError, match="conversation_id or workspace_id"):
            await summaries.create_summary(
                "usr_1",
                {
                    "id": "sum_bad",
                    "conversation_id": None,
                    "workspace_id": None,
                    "source_message_start_seq": None,
                    "source_message_end_seq": None,
                    "summary_kind": "workspace_rollup",
                    "summary_text": "Invalid",
                    "source_object_ids_json": [],
                    "maya_score": 1.5,
                    "model": "score-test-model",
                    "created_at": "2026-04-03T10:00:00+00:00",
                }
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_get_summary_verifies_user_id_via_join() -> None:
    connection, summaries = await _build_runtime()
    try:
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_1",
                "conversation_id": "cnv_1",
                "workspace_id": "wrk_1",
                "source_message_start_seq": 1,
                "source_message_end_seq": 4,
                "summary_kind": "conversation_chunk",
                "summary_text": "Chunk summary",
                "source_object_ids_json": ["mem_1"],
                "maya_score": 1.5,
                "model": "classify-test-model",
                "created_at": "2026-04-03T10:00:00+00:00",
            }
        )

        assert await summaries.get_summary("sum_1", "usr_1") is not None
        assert await summaries.get_summary("sum_1", "usr_2") is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_list_conversation_chunks_returns_ordered_rows() -> None:
    connection, summaries = await _build_runtime()
    try:
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_2",
                "conversation_id": "cnv_1",
                "workspace_id": "wrk_1",
                "source_message_start_seq": 5,
                "source_message_end_seq": 7,
                "summary_kind": "conversation_chunk",
                "summary_text": "Later chunk",
                "source_object_ids_json": [],
                "maya_score": 1.5,
                "model": "classify-test-model",
                "created_at": "2026-04-03T10:02:00+00:00",
            }
        )
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_1",
                "conversation_id": "cnv_1",
                "workspace_id": "wrk_1",
                "source_message_start_seq": 1,
                "source_message_end_seq": 4,
                "summary_kind": "conversation_chunk",
                "summary_text": "Earlier chunk",
                "source_object_ids_json": [],
                "maya_score": 1.5,
                "model": "classify-test-model",
                "created_at": "2026-04-03T10:01:00+00:00",
            }
        )

        rows = await summaries.list_conversation_chunks("usr_1", "cnv_1")

        assert [row["id"] for row in rows] == ["sum_1", "sum_2"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_list_all_conversation_chunks_returns_all_rows_in_order() -> None:
    connection, summaries = await _build_runtime()
    try:
        for index in range(25):
            await summaries.create_summary(
                "usr_1",
                {
                    "id": f"sum_{index:02d}",
                    "conversation_id": "cnv_1",
                    "workspace_id": "wrk_1",
                    "source_message_start_seq": index * 2 + 1,
                    "source_message_end_seq": index * 2 + 2,
                    "summary_kind": "conversation_chunk",
                    "summary_text": f"Chunk {index}",
                    "source_object_ids_json": [],
                    "maya_score": 1.5,
                    "model": "classify-test-model",
                    "created_at": "2026-04-03T10:00:00+00:00",
                }
            )

        rows = await summaries.list_all_conversation_chunks("usr_1", "cnv_1")

        assert len(rows) == 25
        assert rows[0]["id"] == "sum_00"
        assert rows[-1]["id"] == "sum_24"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_workspace_rollups_are_ordered_desc_and_latest_is_returned() -> None:
    connection, summaries = await _build_runtime()
    try:
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_old",
                "conversation_id": None,
                "workspace_id": "wrk_1",
                "source_message_start_seq": None,
                "source_message_end_seq": None,
                "summary_kind": "workspace_rollup",
                "summary_text": "Old rollup",
                "source_object_ids_json": ["mem_1"],
                "maya_score": 1.5,
                "model": "score-test-model",
                "created_at": "2026-04-03T10:01:00+00:00",
            }
        )
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_new",
                "conversation_id": None,
                "workspace_id": "wrk_1",
                "source_message_start_seq": None,
                "source_message_end_seq": None,
                "summary_kind": "workspace_rollup",
                "summary_text": "New rollup",
                "source_object_ids_json": ["mem_2"],
                "maya_score": 1.5,
                "model": "score-test-model",
                "created_at": "2026-04-03T10:02:00+00:00",
            }
        )

        rows = await summaries.list_workspace_rollups("usr_1", "wrk_1")
        latest = await summaries.get_latest_workspace_rollup("usr_1", "wrk_1")

        assert [row["id"] for row in rows] == ["sum_new", "sum_old"]
        assert latest is not None
        assert latest["id"] == "sum_new"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_delete_old_rollups_keeps_n_most_recent() -> None:
    connection, summaries = await _build_runtime()
    try:
        for index in range(4):
            await summaries.create_summary(
                "usr_1",
                {
                    "id": f"sum_{index}",
                    "conversation_id": None,
                    "workspace_id": "wrk_1",
                    "source_message_start_seq": None,
                    "source_message_end_seq": None,
                    "summary_kind": "workspace_rollup",
                    "summary_text": f"Rollup {index}",
                    "source_object_ids_json": [],
                    "maya_score": 1.5,
                    "model": "score-test-model",
                    "created_at": f"2026-04-03T10:0{index}:00+00:00",
                }
            )

        deleted = await summaries.delete_old_rollups("usr_1", "wrk_1", keep_count=2)
        remaining = await summaries.list_workspace_rollups("usr_1", "wrk_1", limit=10)

        assert deleted == 2
        assert [row["id"] for row in remaining] == ["sum_3", "sum_2"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_summary_rejects_wrong_user_ownership() -> None:
    connection, summaries = await _build_runtime()
    try:
        with pytest.raises(ValueError, match="does not belong to user_id"):
            await summaries.create_summary(
                "usr_1",
                {
                    "id": "sum_bad_owner",
                    "conversation_id": "cnv_2",
                    "workspace_id": "wrk_2",
                    "source_message_start_seq": 1,
                    "source_message_end_seq": 2,
                    "summary_kind": "conversation_chunk",
                    "summary_text": "Invalid owner",
                    "source_object_ids_json": [],
                    "maya_score": 1.5,
                    "model": "classify-test-model",
                    "created_at": "2026-04-03T10:00:00+00:00",
                }
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_create_summary_allows_user_level_thematic_profile_rows() -> None:
    connection, summaries = await _build_runtime()
    try:
        summary_id = await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_profile",
                "conversation_id": None,
                "workspace_id": None,
                "source_message_start_seq": None,
                "source_message_end_seq": None,
                "summary_kind": "thematic_profile",
                "hierarchy_level": 2,
                "summary_text": "User consistently prefers incremental debugging and concise checkpoints.",
                "source_object_ids_json": ["mem_1"],
                "maya_score": 1.5,
                "model": "score-test-model",
                "created_at": "2026-04-03T10:00:00+00:00",
            }
        )

        row = await summaries.get_summary(summary_id, "usr_1")

        assert row is not None
        assert row["user_id"] == "usr_1"
        assert row["hierarchy_level"] == 2
        assert row["summary_kind"] == SummaryViewKind.THEMATIC_PROFILE.value
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_delete_summaries_removes_linked_summary_mirrors() -> None:
    connection, summaries = await _build_runtime()
    clock = FrozenClock(datetime(2026, 4, 3, 10, 0, tzinfo=timezone.utc))
    memories = MemoryObjectRepository(connection, clock)
    try:
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_episode",
                "conversation_id": None,
                "workspace_id": None,
                "source_message_start_seq": None,
                "source_message_end_seq": None,
                "summary_kind": "episode",
                "hierarchy_level": 1,
                "summary_text": "Episode summary",
                "source_object_ids_json": ["mem_1"],
                "maya_score": 1.5,
                "model": "score-test-model",
                "created_at": "2026-04-03T10:00:00+00:00",
            }
        )
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_episode",
            summary_kind=SummaryViewKind.EPISODE,
            hierarchy_level=1,
            summary_text="Episode summary",
            source_object_ids=["mem_1"],
            created_at="2026-04-03T10:00:00+00:00",
            scope=MemoryScope.GLOBAL_USER,
        )

        deleted = await summaries.delete_summaries("usr_1", ["sum_episode"])

        assert deleted == 1
        assert await summaries.get_summary("sum_episode", "usr_1") is None
        assert await memories.get_memory_object("sum_mem_sum_episode", "usr_1") is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_delete_summaries_rolls_back_both_deletes_on_failure() -> None:
    connection, summaries = await _build_runtime()
    clock = FrozenClock(datetime(2026, 4, 3, 10, 0, tzinfo=timezone.utc))
    memories = MemoryObjectRepository(connection, clock)
    original_execute = connection.execute
    try:
        await summaries.create_summary(
            "usr_1",
            {
                "id": "sum_episode",
                "conversation_id": None,
                "workspace_id": None,
                "source_message_start_seq": None,
                "source_message_end_seq": None,
                "summary_kind": "episode",
                "hierarchy_level": 1,
                "summary_text": "Episode summary",
                "source_object_ids_json": ["mem_1"],
                "maya_score": 1.5,
                "model": "score-test-model",
                "created_at": "2026-04-03T10:00:00+00:00",
            }
        )
        await memories.upsert_summary_mirror(
            user_id="usr_1",
            summary_view_id="sum_episode",
            summary_kind=SummaryViewKind.EPISODE,
            hierarchy_level=1,
            summary_text="Episode summary",
            source_object_ids=["mem_1"],
            created_at="2026-04-03T10:00:00+00:00",
            scope=MemoryScope.GLOBAL_USER,
        )

        async def failing_execute(query: str, parameters: object = ()) -> object:
            if "DELETE FROM summary_views" in query:
                raise RuntimeError("simulated summary delete failure")
            return await original_execute(query, parameters)

        connection.execute = failing_execute  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="simulated summary delete failure"):
            await summaries.delete_summaries("usr_1", ["sum_episode"])

        assert await summaries.get_summary("sum_episode", "usr_1") is not None
        assert await memories.get_memory_object("sum_mem_sum_episode", "usr_1") is not None
    finally:
        connection.execute = original_execute  # type: ignore[method-assign]
        await connection.close()
