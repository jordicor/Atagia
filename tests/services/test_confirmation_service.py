"""Tests for confirmation-service side effects."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.consent_repository import (
    MemoryConsentProfileRepository,
    PendingMemoryConfirmationRepository,
)
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)
from atagia.services.confirmation_service import PendingConfirmationService
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMCompletionResponse, LLMProvider

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class RecordingEmbeddingIndex:
    vector_limit = 1

    def __init__(self) -> None:
        self.upserts: list[dict[str, object]] = []

    async def upsert(self, memory_id: str, text: str, metadata: dict[str, object]) -> None:
        self.upserts.append(
            {
                "memory_id": memory_id,
                "text": text,
                "metadata": metadata,
            }
        )

    async def search(self, query: str, user_id: str, top_k: int):
        raise AssertionError("search() is not used in confirmation service tests")

    async def delete(self, memory_id: str) -> None:
        raise AssertionError("delete() is not used in confirmation service tests")


class ConfirmationProvider(LLMProvider):
    name = "confirmation-service-tests"

    def __init__(self, intent: str) -> None:
        self.intent = intent
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=f'{{"intent":"{self.intent}"}}',
        )


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        llm_provider="openai",
        llm_api_key=None,
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        llm_base_url=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_extraction_model="extract-test-model",
        llm_scoring_model="score-test-model",
        llm_classifier_model="classify-test-model",
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


@pytest.mark.asyncio
async def test_confirming_pending_memory_upserts_embedding_with_safe_payload() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc))
    embedding_index = RecordingEmbeddingIndex()
    provider = ConfirmationProvider("confirm")
    try:
        await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        confirmations = PendingMemoryConfirmationRepository(connection, clock)
        profiles = MemoryConsentProfileRepository(connection, clock)
        await users.create_user("usr_1")
        await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "personal_assistant",
            "Chat",
        )
        pending = await memories.create_memory_object(
            memory_id="mem_pending",
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="personal_assistant",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Banking card PIN: 4512",
            index_text="bank card PIN",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.97,
            privacy_level=3,
            memory_category=MemoryCategory.PIN_OR_PASSWORD,
            preserve_verbatim=True,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            commit=False,
        )
        await confirmations.create_marker(
            user_id="usr_1",
            conversation_id="cnv_1",
            memory_id=str(pending["id"]),
            category=MemoryCategory.PIN_OR_PASSWORD,
            created_at=str(pending["created_at"]),
            commit=False,
        )
        await confirmations.mark_markers_asked(
            "usr_1",
            [str(pending["id"])],
            asked_at=clock.now().isoformat(),
            commit=False,
        )
        await connection.commit()

        service = PendingConfirmationService(
            connection,
            clock,
            embedding_index,
            llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
            settings=_settings(),
        )
        plan = await service.plan_turn(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="yes",
        )
        await service.apply_turn_plan(user_id="usr_1", plan=plan, commit=True)

        updated = await memories.get_memory_object("mem_pending", "usr_1")
        profile = await profiles.get_profile("usr_1", MemoryCategory.PIN_OR_PASSWORD)
        marker = await confirmations.get_marker_for_memory("usr_1", "mem_pending")

        assert updated is not None
        assert updated["status"] == MemoryStatus.ACTIVE.value
        assert profile is not None
        assert profile["confirmed_count"] == 1
        assert marker is None
        assert embedding_index.upserts == [
            {
                "memory_id": "mem_pending",
                "text": "bank card PIN",
                "metadata": {
                    "user_id": "usr_1",
                    "object_type": "evidence",
                    "scope": "global_user",
                    "created_at": str(pending["created_at"]),
                    "index_text": None,
                },
            }
        ]
        assert provider.requests[0].metadata["purpose"] == "consent_confirmation_intent"
    finally:
        await connection.close()
