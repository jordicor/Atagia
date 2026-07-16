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
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
)
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
)
from atagia.services.confirmation_service import PendingConfirmationService
from atagia.services.errors import (
    TranscriptRebuildInProgressError,
    TranscriptRebuildRemediationRequiredError,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMProvider,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


class RecordingEmbeddingIndex:
    vector_limit = 1

    def __init__(self) -> None:
        self.upserts: list[dict[str, object]] = []

    async def upsert(
        self, memory_id: str, text: str, metadata: dict[str, object]
    ) -> None:
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
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _install_blocking_selection(
    connection: object,
    clock: FrozenClock,
    *,
    state: str,
) -> None:
    now = clock.now().isoformat()
    workflow_id = f"trw_confirmation_{state}"
    await connection.execute(
        """
        INSERT INTO transcript_rebuild_workflows(
            id, operation_id, user_id, conversation_id, selection_epoch,
            transcript_hash, mutation_kind, selected_message_ids_json,
            abandoned_message_ids_json, supporting_message_ids_json,
            affected_memory_ids_json, affected_summary_ids_json,
            orchestrator_job_id, stage, start_derivation_revision,
            created_at, updated_at
        ) VALUES (?, ?, 'usr_1', 'cnv_1', 1, ?, 'replace', '[]', '[]',
                  '[]', '[]', '[]', ?, ?, 0, ?, ?)
        """,
        (
            workflow_id,
            f"op_confirmation_{state}",
            f"hash_confirmation_{state}",
            f"job_confirmation_{state}",
            "remediation_required" if state == "remediation_required" else "aggregates",
            now,
            now,
        ),
    )
    await connection.execute(
        """
        INSERT INTO conversation_transcript_selections(
            user_id, conversation_id, selection_epoch, transcript_hash,
            current_workflow_id, state, updated_at
        ) VALUES ('usr_1', 'cnv_1', 1, ?, ?, ?, ?)
        """,
        (f"hash_confirmation_{state}", workflow_id, state, now),
    )
    await connection.commit()


@pytest.mark.asyncio
async def test_confirming_pending_memory_upserts_embedding_with_safe_payload() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc))
    embedding_index = RecordingEmbeddingIndex()
    provider = ConfirmationProvider("confirm")
    try:
        await sync_assistant_modes(
            connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock
        )
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
            intended_scope=MemoryScope.USER,
            intended_sensitivity=MemorySensitivity.SECRET,
            policy_snapshot={"source": "test"},
            policy_proven=True,
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
                    "scope": "user",
                    "created_at": str(pending["created_at"]),
                    "index_text": None,
                },
            }
        ]
        assert provider.requests[0].metadata["purpose"] == "consent_confirmation_intent"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_unproven_pending_policy_confirmation_moves_to_review_not_active() -> (
    None
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc))
    provider = ConfirmationProvider("confirm")
    try:
        await sync_assistant_modes(
            connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock
        )
        users = UserRepository(connection, clock)
        conversations = ConversationRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        confirmations = PendingMemoryConfirmationRepository(connection, clock)
        await users.create_user("usr_1")
        await conversations.create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "personal_assistant",
            "Chat",
        )
        pending = await memories.create_memory_object(
            memory_id="mem_unproven",
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="personal_assistant",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Sensitive migrated value",
            index_text="sensitive value",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.97,
            privacy_level=3,
            memory_category=MemoryCategory.PIN_OR_PASSWORD,
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
            llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
            settings=_settings(),
        )
        plan = await service.plan_turn(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="yes",
        )
        await service.apply_turn_plan(user_id="usr_1", plan=plan, commit=True)

        updated = await memories.get_memory_object("mem_unproven", "usr_1")
        marker = await confirmations.get_marker_for_memory("usr_1", "mem_unproven")

        assert updated is not None
        assert updated["status"] == MemoryStatus.REVIEW_REQUIRED.value
        assert marker is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_confirmation_rechecks_current_preferences_before_activation() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc))
    provider = ConfirmationProvider("confirm")
    try:
        await sync_assistant_modes(
            connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock
        )
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
            memory_id="mem_current_narrowed",
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="personal_assistant",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            scope_canonical=MemoryScope.USER.value,
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
            intended_scope=MemoryScope.USER,
            intended_sensitivity=MemorySensitivity.SECRET,
            policy_snapshot={"source": "test"},
            policy_proven=True,
            commit=False,
        )
        await confirmations.mark_markers_asked(
            "usr_1",
            [str(pending["id"])],
            asked_at=clock.now().isoformat(),
            commit=False,
        )
        await connection.commit()
        await users.update_memory_preferences("usr_1", remember_across_chats=False)

        service = PendingConfirmationService(
            connection,
            clock,
            llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
            settings=_settings(),
        )
        plan = await service.plan_turn(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="yes",
        )
        await service.apply_turn_plan(user_id="usr_1", plan=plan, commit=True)

        updated = await memories.get_memory_object("mem_current_narrowed", "usr_1")
        profile = await profiles.get_profile("usr_1", MemoryCategory.PIN_OR_PASSWORD)
        marker = await confirmations.get_marker_for_memory(
            "usr_1", "mem_current_narrowed"
        )

        assert updated is not None
        assert updated["status"] == MemoryStatus.REVIEW_REQUIRED.value
        assert profile is None
        assert marker is None
    finally:
        await connection.close()


@pytest.mark.parametrize(
    ("state", "expected_error"),
    [
        ("rebuilding", TranscriptRebuildInProgressError),
        (
            "remediation_required",
            TranscriptRebuildRemediationRequiredError,
        ),
    ],
)
@pytest.mark.parametrize("operation", ["list", "confirm", "decline"])
@pytest.mark.asyncio
async def test_pending_confirmation_public_operations_respect_rebuild_fence(
    state: str,
    expected_error: type[Exception],
    operation: str,
) -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc))
    try:
        await sync_assistant_modes(
            connection,
            ManifestLoader(MANIFESTS_DIR).load_all(),
            clock,
        )
        await UserRepository(connection, clock).create_user("usr_1")
        await ConversationRepository(
            connection,
            clock,
        ).create_conversation(
            "cnv_1",
            "usr_1",
            None,
            "personal_assistant",
            "Chat",
        )
        pending = await MemoryObjectRepository(
            connection,
            clock,
        ).create_memory_object(
            memory_id="mem_blocked_confirmation",
            user_id="usr_1",
            conversation_id="cnv_1",
            assistant_mode_id="personal_assistant",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.USER,
            canonical_text="Sensitive value",
            index_text="sensitive value",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.97,
            privacy_level=3,
            memory_category=MemoryCategory.PIN_OR_PASSWORD,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            commit=False,
        )
        markers = PendingMemoryConfirmationRepository(connection, clock)
        await markers.create_marker(
            user_id="usr_1",
            conversation_id="cnv_1",
            memory_id=str(pending["id"]),
            category=MemoryCategory.PIN_OR_PASSWORD,
            created_at=str(pending["created_at"]),
            commit=False,
        )
        await connection.commit()
        await _install_blocking_selection(connection, clock, state=state)

        service = PendingConfirmationService(connection, clock)
        with pytest.raises(expected_error):
            if operation == "list":
                await service.list_pending_confirmations(user_id="usr_1")
            elif operation == "confirm":
                await service.confirm_pending_memory(
                    user_id="usr_1",
                    memory_id="mem_blocked_confirmation",
                )
            else:
                await service.decline_pending_memory(
                    user_id="usr_1",
                    memory_id="mem_blocked_confirmation",
                )

        unchanged = await MemoryObjectRepository(
            connection,
            clock,
        ).get_memory_object("mem_blocked_confirmation", "usr_1")
        marker = await markers.get_marker_for_memory(
            "usr_1",
            "mem_blocked_confirmation",
        )
        assert unchanged is not None
        assert unchanged["status"] == MemoryStatus.PENDING_USER_CONFIRMATION.value
        assert marker is not None
    finally:
        await connection.close()
