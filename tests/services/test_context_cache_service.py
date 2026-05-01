"""Tests for adaptive context-cache orchestration."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.memory.policy_manifest import compute_effective_policy_hash
from atagia.models.schemas_replay import AblationConfig
from atagia.services.chat_support import (
    default_operational_profile_snapshot,
    resolve_operational_profile,
    resolve_policy,
)
from atagia.services.context_cache_service import ContextCacheService
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
_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')


class ContextCacheProvider(LLMProvider):
    name = "context-cache-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose == "need_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "needs": [],
                        "temporal_range": None,
                        "sub_queries": ["retry loop"],
                        "sparse_query_hints": [
                            {
                                "sub_query_text": "retry loop",
                                "fts_phrase": "retry loop",
                            }
                        ],
                        "query_type": "default",
                        "retrieval_levels": [0],
                    }
                ),
            )
        if purpose == "context_cache_signal_detection":
            prompt = request.messages[1].content
            short_followup = "continue" in prompt.lower()
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "contradiction_detected": False,
                        "high_stakes_topic": False,
                        "sensitive_content": False,
                        "mode_shift_target": None,
                        "short_followup": short_followup,
                        "ambiguous_wording": False,
                    }
                ),
            )
        if purpose == "applicability_scoring":
            memory_ids = _MEMORY_ID_PATTERN.findall(request.messages[1].content)
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    [
                        {"memory_id": memory_id, "llm_applicability": 0.5}
                        for memory_id in memory_ids
                    ]
                ),
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in context cache tests: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-context-cache.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/reply-test-model",
        llm_ingest_model="openai/extract-test-model",
        llm_retrieval_model="openai/score-test-model",
        llm_component_models={"intent_classifier": "openai/classify-test-model"},
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
) -> tuple[AppRuntime, ContextCacheProvider]:
    provider = ContextCacheProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    return runtime, provider


async def _seed_conversation(
    runtime: AppRuntime,
    *,
    user_id: str,
    conversation_id: str,
    assistant_mode_id: str = "coding_debug",
) -> dict[str, object]:
    connection = await runtime.open_connection()
    try:
        users = UserRepository(connection, runtime.clock)
        conversations = ConversationRepository(connection, runtime.clock)
        await users.create_user(user_id)
        return await conversations.create_conversation(
            conversation_id,
            user_id,
            None,
            assistant_mode_id,
            "Chat",
        )
    finally:
        await connection.close()


def _normal_cache_key(
    runtime: AppRuntime,
    service: ContextCacheService,
    conversation: dict[str, object],
) -> str:
    snapshot = default_operational_profile_snapshot(
        loader=runtime.operational_profile_loader,
        settings=runtime.settings,
    )
    return service.build_cache_key(
        user_id=str(conversation["user_id"]),
        assistant_mode_id=str(conversation["assistant_mode_id"]),
        conversation_id=str(conversation["id"]),
        workspace_id=conversation.get("workspace_id"),
        operational_profile_token=snapshot.token,
    )


def _cache_entry_payload(
    runtime: AppRuntime,
    *,
    cache_key: str,
    policy_prompt_hash: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, object]:
    resolved_operational_profile = resolve_operational_profile(
        loader=runtime.operational_profile_loader,
        settings=runtime.settings,
    )
    resolved_policy = resolve_policy(
        runtime.manifests,
        "coding_debug",
        runtime.policy_resolver,
        resolved_operational_profile,
    )
    return {
        "cache_key": cache_key,
        "user_id": "usr_1",
        "conversation_id": "cnv_1",
        "assistant_mode_id": "coding_debug",
        "policy_prompt_hash": policy_prompt_hash or resolved_policy.prompt_hash,
        "effective_policy_hash": compute_effective_policy_hash(resolved_policy),
        "operational_profile": resolved_operational_profile.snapshot.model_dump(mode="json"),
        "workspace_id": workspace_id,
        "composed_context": {
            "contract_block": "",
            "workspace_block": "",
            "memory_block": "",
            "state_block": "",
            "selected_memory_ids": [],
            "total_tokens_estimate": 0,
            "budget_tokens": 500,
            "items_included": 0,
            "items_dropped": 0,
        },
        "contract": {},
        "memory_summaries": [],
        "detected_needs": [],
        "source_retrieval_plan": {},
        "selected_memory_ids": [],
        "cached_at": runtime.clock.now().isoformat(),
        "last_retrieval_message_seq": 1,
        "last_user_message_text": "retry loop",
        "source": "sync",
    }


@pytest.mark.asyncio
async def test_context_cache_service_cache_miss_builds_pending_entry_and_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)
        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Please help me debug this retry loop.",
            )
        finally:
            await connection.close()

        assert resolution.from_cache is False
        assert resolution.cache_source == "sync"
        assert resolution.pending_cache_entry is not None

        published = await service.publish_pending_cache_entry(
            resolution,
            last_retrieval_message_seq=1,
        )
        stored = await runtime.storage_backend.get_context_view(str(resolution.cache_key))

        assert published is True
        assert stored is not None
        assert stored["last_retrieval_message_seq"] == 1
        assert stored["assistant_mode_id"] == "coding_debug"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_cache_hit_skips_need_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)

        connection = await runtime.open_connection()
        try:
            first = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Please help me debug this retry loop.",
            )
        finally:
            await connection.close()
        await service.publish_pending_cache_entry(first, last_retrieval_message_seq=1)
        need_count_before = sum(
            1 for request in provider.requests if request.metadata.get("purpose") == "need_detection"
        )

        connection = await runtime.open_connection()
        try:
            second = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="continue",
            )
        finally:
            await connection.close()

        need_count_after = sum(
            1 for request in provider.requests if request.metadata.get("purpose") == "need_detection"
        )
        assert second.from_cache is True
        assert second.detected_needs == []
        assert second.need_detection_skipped is True
        assert second.pending_cache_entry is None
        assert need_count_after == need_count_before
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_operational_profile_changes_cache_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)

        connection = await runtime.open_connection()
        try:
            normal = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Please help me debug this retry loop.",
            )
        finally:
            await connection.close()
        await service.publish_pending_cache_entry(normal, last_retrieval_message_seq=1)

        connection = await runtime.open_connection()
        try:
            offline = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="continue",
                operational_profile="offline",
            )
        finally:
            await connection.close()

        assert normal.cache_key != offline.cache_key
        assert offline.from_cache is False
        assert offline.resolved_operational_profile.snapshot.profile_id == "offline"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_policy_hash_mismatch_forces_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        conversation = await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)
        cache_key = _normal_cache_key(runtime, service, conversation)
        await runtime.storage_backend.set_context_view(
            cache_key,
            _cache_entry_payload(
                runtime,
                cache_key=cache_key,
                policy_prompt_hash="mismatch",
            ),
            ttl_seconds=30,
        )

        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="continue",
            )
        finally:
            await connection.close()

        assert resolution.from_cache is False
        assert resolution.pending_cache_entry is not None
        assert await runtime.storage_backend.get_context_view(cache_key) is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_effective_policy_hash_mismatch_forces_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        conversation = await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)
        cache_key = _normal_cache_key(runtime, service, conversation)
        payload = _cache_entry_payload(runtime, cache_key=cache_key)
        payload["effective_policy_hash"] = "mismatch"
        await runtime.storage_backend.set_context_view(
            cache_key,
            payload,
            ttl_seconds=30,
        )

        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="continue",
            )
        finally:
            await connection.close()

        assert resolution.from_cache is False
        assert resolution.pending_cache_entry is not None
        assert await runtime.storage_backend.get_context_view(cache_key) is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_workspace_mismatch_forces_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        conversation = await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)
        cache_key = _normal_cache_key(runtime, service, conversation)
        await runtime.storage_backend.set_context_view(
            cache_key,
            _cache_entry_payload(
                runtime,
                cache_key=cache_key,
                workspace_id="wrk_1",
            ),
            ttl_seconds=30,
        )

        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="continue",
            )
        finally:
            await connection.close()

        assert resolution.from_cache is False
        assert resolution.pending_cache_entry is not None
        assert await runtime.storage_backend.get_context_view(cache_key) is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_invalid_cache_entry_is_deleted_before_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        conversation = await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)
        cache_key = _normal_cache_key(runtime, service, conversation)
        await runtime.storage_backend.set_context_view(
            cache_key,
            {"cache_key": cache_key},
            ttl_seconds=30,
        )

        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="continue",
            )
        finally:
            await connection.close()

        assert resolution.from_cache is False
        assert resolution.pending_cache_entry is not None
        assert await runtime.storage_backend.get_context_view(cache_key) is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_monotonic_publish_rejects_older_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)

        connection = await runtime.open_connection()
        try:
            older = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="First request",
            )
            newer = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Second request",
            )
        finally:
            await connection.close()

        assert await service.publish_pending_cache_entry(newer, last_retrieval_message_seq=6) is True
        assert await service.publish_pending_cache_entry(older, last_retrieval_message_seq=5) is False

        stored = await runtime.storage_backend.get_context_view(str(newer.cache_key))
        assert stored is not None
        assert stored["last_user_message_text"] == "Second request"
        assert stored["last_retrieval_message_seq"] == 6
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_ablation_disables_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)

        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Please help me debug this retry loop.",
                ablation=AblationConfig(disable_context_cache=True),
            )
        finally:
            await connection.close()

        assert resolution.from_cache is False
        assert resolution.pending_cache_entry is None
        assert resolution.cache_key is not None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_budgeted_composer_strategy_disables_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)

        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Please help me debug this retry loop.",
                ablation=AblationConfig(composer_strategy="budgeted_marginal"),
            )
        finally:
            await connection.close()

        assert resolution.from_cache is False
        assert resolution.pending_cache_entry is None
        assert resolution.cache_key is not None
    finally:
        await runtime.close()
