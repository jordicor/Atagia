"""Tests for content-language metadata backfill."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository
from atagia.content_language_backfill_cli import build_parser
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind, MemoryStatus
from atagia.services.content_language_backfill_service import (
    ContentLanguageBackfillService,
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


class SequencedLanguageProvider(LLMProvider):
    name = "content-language-backfill-tests"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if not self.payloads:
            raise AssertionError("No canned language payloads remain")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.payloads.pop(0)),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used by content language backfill")


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
    language_codes: list[str] | None = None,
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
        (
            "coding_debug",
            "Coding Debug",
            "stub",
            "{}",
            timestamp,
            timestamp,
        ),
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
        language_codes=language_codes,
    )


def _service(connection, provider: SequencedLanguageProvider) -> ContentLanguageBackfillService:
    return ContentLanguageBackfillService(
        connection=connection,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        settings=_settings(),
    )


def test_content_language_backfill_cli_defaults_to_dry_run() -> None:
    parser = build_parser()
    args = parser.parse_args(["--batch-size", "7", "--user-id", "usr_1"])

    assert args.batch_size == 7
    assert args.user_id == "usr_1"
    assert args.write is False


@pytest.mark.asyncio
async def test_content_language_backfill_dry_run_classifies_without_writing() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedLanguageProvider(
        [{"language_codes": ["en"], "confidence": 0.91}]
    )
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_missing",
            user_id="usr_1",
            canonical_text="Ben moved to a new apartment.",
        )

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            dry_run=True,
        )

        memories = MemoryObjectRepository(connection, clock)
        row = await memories.get_memory_object("mem_missing", "usr_1")
        assert result.model_dump() == {
            "examined": 1,
            "classified": 1,
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "dry_run": True,
            "batch_size": 10,
            "delay_ms": 0,
            "user_id": None,
        }
        assert row is not None
        assert row["language_codes_json"] is None
        prompt = provider.requests[0].messages[1].content
        assert "<canonical_text>" in prompt
        assert "Ben moved to a new apartment." in prompt
        assert "Do not infer the user's language ability or preference." in prompt
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_content_language_backfill_updates_only_missing_rows_for_requested_user() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedLanguageProvider(
        [{"language_codes": ["jp", "ES", "zz", "es"], "confidence": 0.88}]
    )
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_missing",
            user_id="usr_1",
            canonical_text="Rosa toma amlodipino los martes.",
        )
        await _create_memory(
            connection,
            clock,
            memory_id="mem_other_user",
            user_id="usr_2",
            canonical_text="This should not be scanned for usr_1.",
        )
        await _create_memory(
            connection,
            clock,
            memory_id="mem_existing",
            user_id="usr_1",
            canonical_text="Already has metadata.",
            language_codes=["en"],
        )
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
            user_id="usr_1",
            dry_run=False,
        )

        memories = MemoryObjectRepository(connection, clock)
        assert result.examined == 1
        assert result.classified == 1
        assert result.updated == 1
        assert (await memories.get_memory_object("mem_missing", "usr_1"))[
            "language_codes_json"
        ] == ["es"]
        assert (await memories.get_memory_object("mem_existing", "usr_1"))[
            "language_codes_json"
        ] == ["en"]
        assert (await memories.get_memory_object("mem_deleted", "usr_1"))[
            "language_codes_json"
        ] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_content_language_backfill_repairs_existing_invalid_language_arrays() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedLanguageProvider(
        [
            {"language_codes": ["CA"], "confidence": 0.88},
            {"language_codes": ["EN"], "confidence": 0.88},
        ]
    )
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_invalid_existing",
            user_id="usr_1",
            canonical_text="Aquesta memoria esta en catala.",
            language_codes=["en"],
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET language_codes_json = '["jp"]'
            WHERE id = 'mem_invalid_existing'
            """
        )
        await _create_memory(
            connection,
            clock,
            memory_id="mem_scalar_existing",
            user_id="usr_1",
            canonical_text="This legacy metadata is a scalar.",
            language_codes=["en"],
        )
        await connection.execute(
            """
            UPDATE memory_objects
            SET language_codes_json = '"en"'
            WHERE id = 'mem_scalar_existing'
            """
        )
        await connection.commit()

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            user_id="usr_1",
            dry_run=False,
        )

        memories = MemoryObjectRepository(connection, clock)
        row = await memories.get_memory_object(
            "mem_invalid_existing",
            "usr_1",
        )
        scalar_row = await memories.get_memory_object("mem_scalar_existing", "usr_1")
        assert result.examined == 2
        assert result.classified == 2
        assert result.updated == 2
        assert row is not None
        assert row["language_codes_json"] == ["ca"]
        assert scalar_row is not None
        assert scalar_row["language_codes_json"] == ["en"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_content_language_backfill_skips_uncertain_classifications() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    provider = SequencedLanguageProvider(
        [{"language_codes": [], "confidence": 0.12}]
    )
    try:
        await _create_memory(
            connection,
            clock,
            memory_id="mem_uncertain",
            user_id="usr_1",
            canonical_text="???",
        )

        result = await _service(connection, provider).run(
            batch_size=10,
            delay_ms=0,
            dry_run=False,
        )

        row = await MemoryObjectRepository(connection, clock).get_memory_object(
            "mem_uncertain",
            "usr_1",
        )
        assert result.examined == 1
        assert result.classified == 0
        assert result.updated == 0
        assert result.skipped == 1
        assert row is not None
        assert row["language_codes_json"] is None
    finally:
        await connection.close()
