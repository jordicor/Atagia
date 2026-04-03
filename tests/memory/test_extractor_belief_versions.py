"""Focused extractor tests for belief version persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.extractor import MemoryExtractor
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import ExtractionConversationContext
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


class CannedProvider(LLMProvider):
    name = "canned-belief-versioning"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        if request.metadata.get("purpose") == "intent_classifier_explicit":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "is_explicit": True,
                        "reasoning": "Explicit preference for test.",
                    }
                ),
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(self.payload),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used here")


@pytest.mark.asyncio
async def test_extracting_a_belief_also_creates_belief_version_row() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    await conversations.create_conversation("cnv_1", "usr_1", None, "coding_debug", "Debug")
    provider = CannedProvider(
        {
            "evidences": [],
            "beliefs": [
                {
                    "canonical_text": "terse debugging answers",
                    "scope": "assistant_mode",
                    "confidence": 0.78,
                    "source_kind": "inferred",
                    "privacy_level": 1,
                    "payload": {},
                    "claim_key": "response_style.debugging",
                    "claim_value": "terse",
                }
            ],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": False,
        }
    )
    extractor = MemoryExtractor(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        message_repository=messages,
        memory_repository=memories,
        storage_backend=InProcessBackend(),
    )
    resolved_policy = PolicyResolver().resolve(
        ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"],
        None,
        None,
    )
    try:
        message = await messages.create_message(
            "msg_1",
            "cnv_1",
            "user",
            1,
            "I prefer terse debugging answers.",
            8,
            {},
        )

        await extractor.extract(
            message_text=str(message["text"]),
            role="user",
            conversation_context=ExtractionConversationContext(
                user_id="usr_1",
                conversation_id="cnv_1",
                source_message_id="msg_1",
                workspace_id=None,
                assistant_mode_id="coding_debug",
                recent_messages=[],
            ),
            resolved_policy=resolved_policy,
        )

        cursor = await connection.execute(
            """
            SELECT mo.id, bv.version, bv.claim_key, bv.claim_value_json, bv.is_current
            FROM memory_objects AS mo
            JOIN belief_versions AS bv ON bv.belief_id = mo.id
            WHERE mo.user_id = ?
            """,
            ("usr_1",),
        )
        rows = await cursor.fetchall()

        assert len(rows) == 1
        assert rows[0]["version"] == 1
        assert rows[0]["claim_key"] == "response_style.debugging"
        assert json.loads(rows[0]["claim_value_json"]) == "terse"
        assert rows[0]["is_current"] == 1
    finally:
        await connection.close()
