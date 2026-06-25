"""Tests for the coverage-members payload backfill."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from atagia.core import json_utils
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
)
from atagia.coverage_members_backfill_cli import build_parser
from atagia.memory.extraction_cards import _CARD_SYSTEM_PROMPTS
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)
from atagia.services.coverage_members_backfill_service import (
    CoverageMembersBackfillService,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"

_NO_PAYLOAD_LEFT = object()


class SequencedCoverageProvider(LLMProvider):
    """LLM provider returning canned coverage-card lines in order."""

    name = "coverage-members-backfill-tests"

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("No canned coverage payloads remain")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=str(output),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used by coverage members backfill")


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )


async def _create_memory(
    connection,
    clock: FrozenClock,
    *,
    memory_id: str,
    user_id: str,
    canonical_text: str,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    payload: dict[str, Any] | None = None,
) -> None:
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    conversation_id = f"cnv_{user_id}"
    timestamp = clock.now().isoformat()
    if await users.get_user(user_id) is None:
        await users.create_user(user_id)
    await connection.execute(
        """
        INSERT OR IGNORE INTO assistant_modes(
            id,
            display_name,
            prompt_hash,
            memory_policy_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("coding_debug", "Coding Debug", "stub", "{}", timestamp, timestamp),
    )
    await connection.commit()
    if await conversations.get_conversation(conversation_id, user_id) is None:
        await conversations.create_conversation(
            conversation_id,
            user_id,
            None,
            "coding_debug",
            "Chat",
        )
    await memories.create_memory_object(
        user_id=user_id,
        conversation_id=conversation_id,
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=status,
        memory_id=memory_id,
        payload=payload,
    )


async def _set_payload_key(connection, memory_id: str, payload: dict[str, Any]) -> None:
    """Force a payload_json value (e.g. a row already carrying the key)."""
    await connection.execute(
        "UPDATE memory_objects SET payload_json = ? WHERE id = ?",
        (json_utils.dumps(payload, sort_keys=True), memory_id),
    )
    await connection.commit()


async def _payload(connection, clock: FrozenClock, memory_id: str, user_id: str) -> dict[str, Any]:
    row = await MemoryObjectRepository(connection, clock).get_memory_object(memory_id, user_id)
    assert row is not None
    return row["payload_json"]


def _service(connection, provider: SequencedCoverageProvider) -> CoverageMembersBackfillService:
    return CoverageMembersBackfillService(
        connection=connection,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        settings=_settings(),
    )


def test_coverage_members_backfill_cli_defaults_to_dry_run() -> None:
    parser = build_parser()
    args = parser.parse_args(["--batch-size", "7", "--user-id", "usr_1"])

    assert args.batch_size == 7
    assert args.user_id == "usr_1"
    assert args.write is False


@pytest.mark.asyncio
async def test_scan_selects_only_key_absent_rows() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedCoverageProvider(
        ['cand_001 | [{"member_key": "dr. a", "display_text": "Dr. A"}]']
    )
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_absent",
            user_id="usr_1",
            canonical_text="Rosa sees Dr. A.",
        )
        # Already processed with an empty list: MUST NOT be re-processed.
        await _create_memory(
            connection,
            clock,
            memory_id="mem_empty_list",
            user_id="usr_1",
            canonical_text="Rosa prefers short replies.",
            payload={"coverage_members": []},
        )
        # Already processed with members: MUST NOT be re-processed.
        await _create_memory(
            connection,
            clock,
            memory_id="mem_with_members",
            user_id="usr_1",
            canonical_text="Rosa sees Dr. B.",
            payload={"coverage_members": [{"member_key": "dr. b", "display_text": "Dr. B"}]},
        )

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            dry_run=False,
        )

        assert result.examined == 1
        assert result.processed == 1
        assert result.updated == 1
        # Exactly one LLM call: only the key-absent row was scanned.
        assert len(provider.requests) == 1
        assert (await _payload(connection, clock, "mem_absent", "usr_1"))[
            "coverage_members"
        ] == [{"member_key": "dr. a", "display_text": "Dr. A"}]
        # The pre-existing empty-list row was left untouched.
        assert (await _payload(connection, clock, "mem_empty_list", "usr_1"))[
            "coverage_members"
        ] == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_real_run_writes_key_and_increments_counters() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedCoverageProvider(
        [
            'cand_001 | [{"member_key": "dr. mendez", "display_text": "Dr. Mendez"},'
            ' {"member_key": "dr. patel", "display_text": "Dr. Patel"}]',
            "cand_001 | []",
        ]
    )
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_members",
            user_id="usr_1",
            canonical_text="Rosa sees Dr. Mendez and Dr. Patel.",
            payload={"existing_key": "kept"},
        )
        await _create_memory(
            connection,
            clock,
            memory_id="mem_no_members",
            user_id="usr_1",
            canonical_text="Rosa prefers short replies.",
        )

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            dry_run=False,
        )

        assert result.examined == 2
        assert result.processed == 2
        assert result.updated == 2
        assert result.failed == 0
        members_payload = await _payload(connection, clock, "mem_members", "usr_1")
        assert members_payload["coverage_members"] == [
            {"member_key": "dr. mendez", "display_text": "Dr. Mendez"},
            {"member_key": "dr. patel", "display_text": "Dr. Patel"},
        ]
        # Existing payload keys are preserved.
        assert members_payload["existing_key"] == "kept"
        # Empty-member rows still get the key (processed marker).
        assert (await _payload(connection, clock, "mem_no_members", "usr_1"))[
            "coverage_members"
        ] == []
        # The card purpose is wired correctly on each request.
        assert all(
            req.metadata["purpose"] == "memory_extraction_coverage_members_card"
            for req in provider.requests
        )
        assert all(
            req.messages[0].content == _CARD_SYSTEM_PROMPTS["coverage_members"]
            for req in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_dry_run_writes_nothing() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedCoverageProvider(
        ['cand_001 | [{"member_key": "dr. a", "display_text": "Dr. A"}]']
    )
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_absent",
            user_id="usr_1",
            canonical_text="Rosa sees Dr. A.",
        )

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            dry_run=True,
        )

        assert result.examined == 1
        assert result.processed == 1
        assert result.updated == 0
        # Key remains absent: row is still re-runnable.
        assert "coverage_members" not in (
            await _payload(connection, clock, "mem_absent", "usr_1")
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_per_row_failure_counted_and_leaves_row_rerunnable() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedCoverageProvider(
        [
            RuntimeError("provider blew up on this row"),
            'cand_001 | [{"member_key": "dr. b", "display_text": "Dr. B"}]',
        ]
    )
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_fails",
            user_id="usr_1",
            canonical_text="Rosa sees Dr. A.",
        )
        await _create_memory(
            connection,
            clock,
            memory_id="mem_ok",
            user_id="usr_1",
            canonical_text="Rosa sees Dr. B.",
        )

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            dry_run=False,
        )

        assert result.examined == 2
        assert result.processed == 1
        assert result.updated == 1
        assert result.failed == 1
        # The failed row keeps NO key, so a re-run will pick it up again.
        assert "coverage_members" not in (
            await _payload(connection, clock, "mem_fails", "usr_1")
        )
        assert (await _payload(connection, clock, "mem_ok", "usr_1"))[
            "coverage_members"
        ] == [{"member_key": "dr. b", "display_text": "Dr. B"}]
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card_output",
    [
        "cand_001 | [{]",
        'cand_001 | [{"member_key": "dr. a", "display_text": "Dr. A"}, 7]',
    ],
)
async def test_malformed_card_output_fails_and_leaves_row_rerunnable(
    card_output: str,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedCoverageProvider([card_output])
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_malformed",
            user_id="usr_1",
            canonical_text="Rosa sees Dr. A.",
        )

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            dry_run=False,
        )

        assert result.examined == 1
        assert result.processed == 0
        assert result.updated == 0
        assert result.failed == 1
        assert len(provider.requests) == 1
        assert "coverage_members" not in (
            await _payload(connection, clock, "mem_malformed", "usr_1")
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_user_id_scoping_does_not_touch_other_users() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedCoverageProvider(
        ['cand_001 | [{"member_key": "dr. a", "display_text": "Dr. A"}]']
    )
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_usr_1",
            user_id="usr_1",
            canonical_text="Rosa sees Dr. A.",
        )
        await _create_memory(
            connection,
            clock,
            memory_id="mem_usr_2",
            user_id="usr_2",
            canonical_text="Someone else's row must not be scanned.",
        )

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            user_id="usr_1",
            dry_run=False,
        )

        assert result.examined == 1
        assert result.updated == 1
        assert result.user_id == "usr_1"
        assert len(provider.requests) == 1
        assert (await _payload(connection, clock, "mem_usr_1", "usr_1"))[
            "coverage_members"
        ] == [{"member_key": "dr. a", "display_text": "Dr. A"}]
        # The other user's row was never processed.
        assert "coverage_members" not in (
            await _payload(connection, clock, "mem_usr_2", "usr_2")
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_deleted_rows_are_not_backfilled() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedCoverageProvider([])
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_deleted",
            user_id="usr_1",
            canonical_text="Deleted rows are not backfilled.",
            status=MemoryStatus.DELETED,
        )

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            dry_run=False,
        )

        assert result.examined == 0
        assert result.processed == 0
        assert len(provider.requests) == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_invalid_args_fail_fast() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    provider = SequencedCoverageProvider([])
    try:
        service = _service(connection, provider)
        with pytest.raises(ValueError):
            await service.run(batch_size=0, delay_ms=0)
        with pytest.raises(ValueError):
            await service.run(batch_size=10, delay_ms=-1)
    finally:
        await connection.close()
