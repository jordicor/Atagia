"""Tests for sidecar request/response memory orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageBlocks,
    InitialContextPackageKey,
    InitialContextPackageKind,
)
from atagia.models.schemas_jobs import JobType
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.sidecar_service import SidecarService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


class NoLLMProvider(LLMProvider):
    name = "sidecar-no-llm-tests"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError(f"Sidecar test should not call LLM: {request.metadata}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Sidecar test should not embed: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-sidecar-service.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="reply-test-model",
        llm_forced_global_model="openai/reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        small_corpus_token_threshold_ratio=0.0,
    )


async def _build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AppRuntime:
    provider = NoLLMProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    return await initialize_runtime(_settings(tmp_path))


async def _job_counts(runtime: AppRuntime) -> dict[str, int]:
    connection = await runtime.open_connection()
    try:
        cursor = await connection.execute(
            """
            SELECT job_type, COUNT(*) AS count
            FROM worker_job_runs
            GROUP BY job_type
            """
        )
        return {str(row["job_type"]): int(row["count"]) for row in await cursor.fetchall()}
    finally:
        await connection.close()


async def _upsert_conversation_package(runtime: AppRuntime) -> str:
    connection = await runtime.open_connection()
    try:
        key = InitialContextPackageKey(
            version=2,
            package_kind=InitialContextPackageKind.CONVERSATION,
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
        )
        package = await InitialContextPackageRepository(
            connection,
            runtime.clock,
        ).upsert_package(
            package_kind=key.package_kind,
            version=key.version,
            user_id=key.user_id,
            conversation_id=key.conversation_id,
            retrieval_profile_id=key.retrieval_profile_id,
            key_json=key,
            blocks_json=InitialContextPackageBlocks(
                conversation_summary_block="Prepared context to invalidate.",
            ),
        )
        return package.package_key_hash
    finally:
        await connection.close()


async def _package_status(runtime: AppRuntime, package_key_hash: str) -> str:
    connection = await runtime.open_connection()
    try:
        result = await InitialContextPackageRepository(
            connection,
            runtime.clock,
        ).read_by_key_hash(
            user_id="usr_1",
            package_key_hash=package_key_hash,
        )
        return result.status
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_sidecar_defers_user_jobs_until_response_and_dedupes_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        sidecar = SidecarService(runtime)

        context = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please remember that I prefer concise answers.",
            mode="coding_debug",
            message_id="host-user-1",
        )

        assert context.request_message_id == "host-user-1"
        assert await _job_counts(runtime) == {}

        await sidecar.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Got it.",
            message_id="host-assistant-1",
        )

        assert await _job_counts(runtime) == {
            JobType.EXTRACT_MEMORY_CANDIDATES.value: 2,
            JobType.PROJECT_CONTRACT.value: 1,
            JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value: 1,
        }

        await sidecar.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Got it.",
            message_id="host-assistant-1",
        )

        assert await _job_counts(runtime) == {
            JobType.EXTRACT_MEMORY_CANDIDATES.value: 2,
            JobType.PROJECT_CONTRACT.value: 1,
            JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value: 1,
        }
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_set_memory_preferences_marks_initial_context_packages_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        sidecar = SidecarService(runtime)
        await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Prepare a conversation.",
            mode="coding_debug",
        )
        package_key_hash = await _upsert_conversation_package(runtime)

        await sidecar.set_memory_preferences(
            "usr_1",
            remember_across_chats=False,
        )

        assert await _package_status(runtime, package_key_hash) == "stale"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_set_conversation_incognito_marks_initial_context_packages_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        sidecar = SidecarService(runtime)
        await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Prepare a conversation.",
            mode="coding_debug",
        )
        package_key_hash = await _upsert_conversation_package(runtime)

        await sidecar.set_conversation_incognito(
            "usr_1",
            "cnv_1",
            True,
        )

        assert await _package_status(runtime, package_key_hash) == "stale"
    finally:
        await runtime.close()
