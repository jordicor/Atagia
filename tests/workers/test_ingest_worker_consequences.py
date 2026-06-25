"""Tests for consequence-chain handling in the ingest worker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    MessageRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_jobs import EXTRACT_STREAM_NAME, JobEnvelope
from atagia.models.schemas_memory import MemoryObjectType
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.workers.ingest_worker import IngestWorker
from tests.extraction_payload_support import (
    is_memory_extraction_card_purpose,
    memory_extraction_card_output_from_payload,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class QueueProvider(LLMProvider):
    name = "ingest-consequence-tests"

    def __init__(
        self,
        *,
        extraction_output: str,
        consequence_outputs: dict[str, str] | None = None,
        tendency_output: str | None = None,
        fail_on_consequence_detection: bool = False,
    ) -> None:
        self.extraction_output = extraction_output
        self.consequence_outputs = dict(consequence_outputs or {})
        self.tendency_output = tendency_output
        self.fail_on_consequence_detection = fail_on_consequence_detection
        self.requests: list[LLMCompletionRequest] = []
        self._active_extraction_output: str | None = None

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = request.metadata.get("purpose")
        if is_memory_extraction_card_purpose(purpose):
            if purpose == "memory_extraction_candidate_card":
                self._active_extraction_output = self.extraction_output
            output_text = memory_extraction_card_output_from_payload(
                self._active_extraction_output or self.extraction_output,
                purpose,
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=output_text,
            )
        if (
            isinstance(purpose, str)
            and purpose.startswith("consequence_")
            and purpose.endswith("_card")
        ):
            if self.fail_on_consequence_detection:
                raise RuntimeError("synthetic consequence detector failure")
            output_text = self.consequence_outputs.get(purpose)
            if output_text is None:
                raise AssertionError(f"No consequence output configured for {purpose}")
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=output_text,
            )
        if purpose == "consequence_tendency_inference":
            if self.tendency_output is None:
                raise AssertionError("No tendency output configured")
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=self.tendency_output,
            )
        raise AssertionError(f"Unexpected purpose {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in ingest consequence tests")


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


def _consequence_card_outputs(
    *,
    gate: str = "yes",
    action: str = "Suggested a large refactor.",
    outcome: str = "Regressions appeared afterwards.",
    sentiment: str = "bad",
    link: str = "msg_assistant_1",
    language: str = "en",
) -> dict[str, str]:
    return {
        "consequence_gate_card": gate,
        "consequence_action_card": action,
        "consequence_outcome_card": outcome,
        "consequence_sentiment_card": sentiment,
        "consequence_link_card": link,
        "consequence_language_card": language,
    }


async def _build_runtime(
    *,
    extraction_output: str,
    consequence_outputs: dict[str, str] | None = None,
    tendency_output: str | None = None,
    fail_on_consequence_detection: bool = False,
    role: str = "user",
    message_text: str = "That broke everything after your refactor suggestion.",
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 2, 16, 0, tzinfo=timezone.utc))
    manifest_loader = ManifestLoader(MANIFESTS_DIR)
    await sync_assistant_modes(connection, manifest_loader.load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    messages = MessageRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation(
        "cnv_1", "usr_1", "wrk_1", "coding_debug", "Chat"
    )
    await messages.create_message(
        "msg_assistant_1",
        "cnv_1",
        "assistant",
        1,
        "Try a large refactor to simplify the code path.",
        9,
        {},
    )
    current = await messages.create_message(
        "msg_user_1",
        "cnv_1",
        role,
        2,
        message_text,
        10,
        {},
    )
    backend = InProcessBackend()
    provider = QueueProvider(
        extraction_output=extraction_output,
        consequence_outputs=consequence_outputs,
        tendency_output=tendency_output,
        fail_on_consequence_detection=fail_on_consequence_detection,
    )
    worker = IngestWorker(
        storage_backend=backend,
        connection=connection,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=clock,
        manifest_loader=manifest_loader,
        settings=_settings(),
    )
    return connection, backend, memories, worker, current


def _extract_job(message_id: str, *, role: str, message_text: str) -> JobEnvelope:
    return JobEnvelope(
        job_id="job_extract_consequence_1",
        job_type="extract_memory_candidates",
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=[message_id],
        payload={
            "message_id": message_id,
            "message_text": message_text,
            "role": role,
            "assistant_mode_id": "coding_debug",
            "workspace_id": "wrk_1",
            "recent_messages": [],
        },
        created_at=datetime(2026, 4, 2, 16, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_ingest_worker_detects_consequence_signal_and_builds_chain() -> None:
    extraction_output = json.dumps(
        {
            "evidences": [],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": True,
        }
    )
    tendency_output = json.dumps(
        {"tendency_text": "Prefer incremental patches in this workspace."}
    )
    connection, backend, memories, worker, current = await _build_runtime(
        extraction_output=extraction_output,
        consequence_outputs=_consequence_card_outputs(),
        tendency_output=tendency_output,
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(
                str(current["id"]), role="user", message_text=str(current["text"])
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        stored = await memories.list_for_user("usr_1")
        chains = [
            row
            for row in stored
            if row["object_type"] == MemoryObjectType.CONSEQUENCE_CHAIN.value
        ]
        cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM consequence_chains"
        )
        row = await cursor.fetchone()

        assert result.acked == 1
        assert row["count"] == 1
        assert len(chains) == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_skips_consequence_detection_for_assistant_messages() -> (
    None
):
    extraction_output = json.dumps(
        {
            "evidences": [],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": True,
        }
    )
    connection, backend, _memories, worker, current = await _build_runtime(
        extraction_output=extraction_output,
        consequence_outputs=None,
        tendency_output=None,
        role="assistant",
        message_text="Try a large refactor to simplify the code path.",
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(
                str(current["id"]), role="assistant", message_text=str(current["text"])
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM consequence_chains"
        )
        row = await cursor.fetchone()

        assert result.acked == 1
        assert row["count"] == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_handles_consequence_detector_failure_gracefully() -> None:
    extraction_output = json.dumps(
        {
            "evidences": [],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": True,
        }
    )
    connection, backend, _memories, worker, current = await _build_runtime(
        extraction_output=extraction_output,
        fail_on_consequence_detection=True,
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(
                str(current["id"]), role="user", message_text=str(current["text"])
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM consequence_chains"
        )
        row = await cursor.fetchone()

        assert result.acked == 1
        assert row["count"] == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_ingest_worker_skips_incomplete_consequence_card_outputs() -> None:
    extraction_output = json.dumps(
        {
            "evidences": [],
            "beliefs": [],
            "contract_signals": [],
            "state_updates": [],
            "mode_guess": None,
            "nothing_durable": True,
        }
    )
    connection, backend, _memories, worker, current = await _build_runtime(
        extraction_output=extraction_output,
        consequence_outputs=_consequence_card_outputs(outcome="none"),
        tendency_output=json.dumps({"tendency_text": "Unused tendency."}),
    )
    try:
        await backend.stream_add(
            EXTRACT_STREAM_NAME,
            _extract_job(
                str(current["id"]), role="user", message_text=str(current["text"])
            ).model_dump(mode="json"),
        )

        result = await worker.run_once()
        cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM consequence_chains"
        )
        row = await cursor.fetchone()

        assert result.acked == 1
        assert row["count"] == 0
    finally:
        await connection.close()
