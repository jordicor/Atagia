"""Tests for adaptive context-cache orchestration."""

from __future__ import annotations

import asyncio
import json
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.canonical import canonical_json_bytes
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.memory.policy_manifest import compute_effective_policy_hash
from atagia.models.schemas_memory import ComposedContext, ResponseMode, RetrievalPlan
from atagia.models.schemas_replay import AblationConfig, PipelineResult
from atagia.services.chat_support import (
    default_operational_profile_snapshot,
    resolve_operational_profile,
    resolve_policy,
)
from atagia.core.repositories import MemoryObjectRepository
from atagia.services.context_cache_service import (
    CONTEXT_CACHE_KEY_VERSION,
    ContextCacheService,
)
from atagia.services.prompt_authority import (
    effective_allow_private_for_sql_repository,
    normalize_request_authority_context,
    privacy_sql_filters_disabled,
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
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


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
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(request.messages[1].content)
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "scores": [
                            {"score_key": score_key, "llm_applicability": 0.5}
                            for _memory_id, score_key in candidate_keys
                        ]
                    }
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


def test_cache_key_includes_effective_authority_context() -> None:
    base = {
        "user_id": "user-1",
        "assistant_mode_id": "general_qa",
        "conversation_id": "conv-1",
        "workspace_id": "workspace-1",
        "operational_profile_token": "profile-token",
    }

    enforce_key = ContextCacheService.build_cache_key(
        **base,
        privacy_enforcement="enforce",
        authenticated_user_privilege_level="standard",
        authenticated_user_is_atagia_master=False,
    )
    off_key = ContextCacheService.build_cache_key(
        **base,
        privacy_enforcement="off",
        authenticated_user_privilege_level="standard",
        authenticated_user_is_atagia_master=False,
    )
    master_key = ContextCacheService.build_cache_key(
        **base,
        privacy_enforcement="off",
        authenticated_user_privilege_level="atagia_master",
        authenticated_user_is_atagia_master=True,
    )

    assert enforce_key != off_key
    assert off_key != master_key


def test_normal_response_mode_cache_key_is_byte_identical_to_pre_change() -> None:
    """Normal-mode keys must not shift when the response_mode arg is added.

    The expected subject below is exactly what the key builder produced before
    fast modes existed (no ``response_mode`` field). Default and explicit normal
    must both match that hash byte-for-byte so no cache is invalidated.
    """
    base = {
        "user_id": "user-1",
        "assistant_mode_id": "general_qa",
        "conversation_id": "conv-1",
        "workspace_id": "workspace-1",
        "operational_profile_token": "profile-token",
    }
    pre_change_subject = {
        "v": CONTEXT_CACHE_KEY_VERSION,
        "active_embodiment_id": None,
        "active_mind_id": None,
        "active_presence_id": None,
        "active_realm_id": None,
        "active_space_id": None,
        "assistant_mode_id": "general_qa",
        "conversation_id": "conv-1",
        "mind_topology": "unimind",
        "operational_profile_token": "profile-token",
        "privacy_enforcement": "enforce",
        "authenticated_user_privilege_level": "standard",
        "authenticated_user_is_atagia_master": False,
        "user_id": "user-1",
        "workspace_id": "workspace-1",
    }
    pre_change_key = (
        f"ctx:v{CONTEXT_CACHE_KEY_VERSION}:"
        + hashlib.sha256(canonical_json_bytes(pre_change_subject)).hexdigest()
    )

    default_key = ContextCacheService.build_cache_key(**base)
    explicit_normal_key = ContextCacheService.build_cache_key(
        **base, response_mode=ResponseMode.NORMAL
    )

    assert default_key == pre_change_key
    assert explicit_normal_key == pre_change_key


def test_response_mode_partitions_cache_key_space() -> None:
    base = {
        "user_id": "user-1",
        "assistant_mode_id": "general_qa",
        "conversation_id": "conv-1",
        "workspace_id": "workspace-1",
        "operational_profile_token": "profile-token",
    }
    normal_key = ContextCacheService.build_cache_key(**base)
    fast_key = ContextCacheService.build_cache_key(
        **base, response_mode=ResponseMode.FAST
    )
    smart_fast_key = ContextCacheService.build_cache_key(
        **base, response_mode=ResponseMode.SMART_FAST
    )

    assert len({normal_key, fast_key, smart_fast_key}) == 3


def test_guard_retrieval_diagnostics_include_answer_evidence() -> None:
    composed_context = ComposedContext(
        memory_block="[Retrieved Memories]\n1. related context",
        selected_memory_ids=["mem_vibes"],
        answer_evidence_sufficiency={
            "state": "sufficient_direct_quote",
            "confidence": 0.91,
            "rendered": False,
            "top_memory_id": "mem_vibes",
        },
        answer_evidence_items=[
            {
                "memory_id": "mem_vibes",
                "claim": "Jon wanted to savor the good vibes.",
                "supporting_quote": "Jon: I want to savor all the good vibes.",
                "quote_source": "source_message",
                "support_kind": "contextual_direct",
                "source_chain": [
                    "assistant seq 282: The studio looks amazing.",
                    "user seq 283: I want to savor all the good vibes.",
                ],
                "selected_for_answer_pack": False,
                "final_score": 0.94,
            }
        ],
        answer_shape="list",
        coverage_mode="exhaustive_known_set",
        source_precision="required",
        coverage_state="complete",
        allowed_values=[
            {
                "display_text": "good vibes",
                "normalized_key": "value|good vibes",
                "evidence_ids": ["memory:mem_vibes"],
            }
        ],
        total_tokens_estimate=18,
        budget_tokens=100,
        items_included=1,
        items_dropped=0,
    )
    result = PipelineResult(
        retrieval_plan=RetrievalPlan(
            assistant_mode_id="coding_debug",
            conversation_id="cnv_1",
            max_candidates=5,
            max_context_items=4,
            privacy_ceiling=3,
            privacy_enforcement="off",
        ),
        composed_context=composed_context,
    )

    diagnostics = ContextCacheService._guard_retrieval_diagnostics(result)

    assert diagnostics["answer_evidence"]["sufficiency"]["state"] == (
        "sufficient_direct_quote"
    )
    assert diagnostics["answer_evidence"]["direct_memory_ids"] == ["mem_vibes"]
    assert diagnostics["answer_evidence"]["items"][0]["supporting_quote"] == (
        "Jon: I want to savor all the good vibes."
    )
    assert diagnostics["answer_support"]["allowed_values"][0]["display_text"] == (
        "good vibes"
    )
    assert diagnostics["answer_support"]["coverage_state"] == "complete"


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
        active_presence_id=conversation.get("active_presence_id"),
        active_space_id=conversation.get("active_space_id"),
        operational_profile_token=snapshot.token,
    )


def _legacy_v4_cache_key(
    runtime: AppRuntime,
    conversation: dict[str, object],
) -> str:
    snapshot = default_operational_profile_snapshot(
        loader=runtime.operational_profile_loader,
        settings=runtime.settings,
    )
    cache_subject = {
        "v": 4,
        "active_presence_id": conversation.get("active_presence_id"),
        "assistant_mode_id": str(conversation["assistant_mode_id"]),
        "conversation_id": str(conversation["id"]),
        "operational_profile_token": snapshot.token,
        "user_id": str(conversation["user_id"]),
        "workspace_id": conversation.get("workspace_id"),
    }
    return "ctx:v4:" + hashlib.sha256(canonical_json_bytes(cache_subject)).hexdigest()


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
        assert (
            resolution.retrieval_diagnostics_for_guard["need_detection"]["query_type"]
            == "default"
        )
        assert (
            resolution.retrieval_diagnostics_for_guard[
                "diagnostic_shape_fallback_used"
            ]
            is False
        )
        assert (
            resolution.pending_cache_entry.retrieval_diagnostics_for_guard
            == resolution.retrieval_diagnostics_for_guard
        )

        published = await service.publish_pending_cache_entry(
            resolution,
            last_retrieval_message_seq=1,
        )
        stored = await runtime.storage_backend.get_context_view(str(resolution.cache_key))

        assert published is True
        assert stored is not None
        assert stored["last_retrieval_message_seq"] == 1
        assert stored["assistant_mode_id"] == "coding_debug"
        assert stored["retrieval_diagnostics_for_guard"]["need_detection"][
            "query_type"
        ] == "default"
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
        assert second.retrieval_diagnostics_for_guard["need_detection"][
            "query_type"
        ] == "default"
        assert need_count_after == need_count_before
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_ignores_legacy_v4_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        conversation = await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)
        legacy_cache_key = _legacy_v4_cache_key(runtime, conversation)
        await runtime.storage_backend.set_context_view(
            legacy_cache_key,
            _cache_entry_payload(runtime, cache_key=legacy_cache_key),
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
        assert resolution.cache_key is not None
        assert str(resolution.cache_key).startswith(f"ctx:v{CONTEXT_CACHE_KEY_VERSION}:")
        assert resolution.cache_key != legacy_cache_key
        assert resolution.pending_cache_entry is not None
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


@pytest.mark.asyncio
async def test_context_cache_service_active_presence_changes_cache_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        conversation = await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)
        snapshot = default_operational_profile_snapshot(
            loader=runtime.operational_profile_loader,
            settings=runtime.settings,
        )

        first = service.build_cache_key(
            user_id=str(conversation["user_id"]),
            assistant_mode_id=str(conversation["assistant_mode_id"]),
            conversation_id=str(conversation["id"]),
            workspace_id=conversation.get("workspace_id"),
            active_presence_id="presence_alpha",
            operational_profile_token=snapshot.token,
        )
        second = service.build_cache_key(
            user_id=str(conversation["user_id"]),
            assistant_mode_id=str(conversation["assistant_mode_id"]),
            conversation_id=str(conversation["id"]),
            workspace_id=conversation.get("workspace_id"),
            active_presence_id="presence_beta",
            operational_profile_token=snapshot.token,
        )

        assert first != second
        assert first.startswith(f"ctx:v{CONTEXT_CACHE_KEY_VERSION}:")
        assert second.startswith(f"ctx:v{CONTEXT_CACHE_KEY_VERSION}:")
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_active_space_changes_cache_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        conversation = await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
        service = ContextCacheService(runtime)
        snapshot = default_operational_profile_snapshot(
            loader=runtime.operational_profile_loader,
            settings=runtime.settings,
        )

        first = service.build_cache_key(
            user_id=str(conversation["user_id"]),
            assistant_mode_id=str(conversation["assistant_mode_id"]),
            conversation_id=str(conversation["id"]),
            workspace_id=conversation.get("workspace_id"),
            active_presence_id=conversation.get("active_presence_id"),
            active_space_id="space_alpha",
            operational_profile_token=snapshot.token,
        )
        second = service.build_cache_key(
            user_id=str(conversation["user_id"]),
            assistant_mode_id=str(conversation["assistant_mode_id"]),
            conversation_id=str(conversation["id"]),
            workspace_id=conversation.get("workspace_id"),
            active_presence_id=conversation.get("active_presence_id"),
            active_space_id="space_beta",
            operational_profile_token=snapshot.token,
        )

        assert first != second
        assert first.startswith(f"ctx:v{CONTEXT_CACHE_KEY_VERSION}:")
        assert second.startswith(f"ctx:v{CONTEXT_CACHE_KEY_VERSION}:")
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_context_cache_service_retrieval_overrides_disable_cache(
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
            _cache_entry_payload(runtime, cache_key=cache_key),
            ttl_seconds=30,
        )

        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="continue",
                ablation=AblationConfig(
                    override_retrieval_params={
                        "privacy_ceiling": 3,
                        "allow_private_sensitivity": True,
                    }
                ),
            )
        finally:
            await connection.close()

        assert resolution.from_cache is False
        assert resolution.pending_cache_entry is None
        assert resolution.cache_key == cache_key
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_fast_contract_lookup_master_enforce_matches_normal_sql_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast contract lookup must gate sensitivity exactly like the normal pipeline.

    For an atagia_master authority running with privacy_enforcement="enforce",
    master must NOT silently widen the SQL sensitivity gate: the fast path has
    to pass the same (gates_enabled, allow_private) pair the full pipeline
    computes, and secret-tier rows must stay out of the SQL clause.
    """
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        conversation = await _seed_conversation(
            runtime, user_id="usr_master", conversation_id="cnv_master"
        )
        service = ContextCacheService(runtime)

        captured: dict[str, object] = {}

        async def capture_contract(self, *args, **kwargs):
            captured.update(kwargs)
            return {}

        monkeypatch.setattr(
            "atagia.memory.contract_projection.ContractProjector.get_current_contract",
            capture_contract,
        )

        resolved_operational_profile = resolve_operational_profile(
            loader=runtime.operational_profile_loader,
            settings=runtime.settings,
        )
        resolved_policy = resolve_policy(
            runtime.manifests,
            str(conversation["assistant_mode_id"]),
            runtime.policy_resolver,
            resolved_operational_profile,
        )
        authority_context = normalize_request_authority_context(
            privacy_enforcement="enforce",
            authenticated_user_is_atagia_master=True,
            user_id="usr_master",
            purpose="context_cache_fast",
        )
        ablation = AblationConfig(privacy_enforcement="enforce")

        connection = await runtime.open_connection()
        try:
            await service._fast_contract_lookup(
                connection,
                conversation=conversation,
                resolved_policy=resolved_policy,
                authority_context=authority_context,
                ablation=ablation,
            )
        finally:
            await connection.close()

        expected_gates = privacy_sql_filters_disabled(ablation)
        expected_allow_private = effective_allow_private_for_sql_repository(
            resolved_policy, ablation
        )
        assert captured["sensitivity_gates_enabled"] == expected_gates
        assert captured["allow_private_sensitivity"] == expected_allow_private
        # Master + enforce keeps the gate closed; the relaxed (gates_enabled=True)
        # branch that admits secret rows must not be taken.
        assert captured["sensitivity_gates_enabled"] is False
        clause = MemoryObjectRepository.sensitivity_filter_clause(
            gates_enabled=bool(captured["sensitivity_gates_enabled"]),
            allow_private_sensitivity=bool(captured["allow_private_sensitivity"]),
        )
        assert "secret" not in clause
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_smart_fast_warm_holds_guard_only_around_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow warm resolve must not block a same-user foreground guard acquire.

    The per-user cache guard is held only around the cache publish, so while the
    warm's (slow) retrieval resolve is in flight a foreground turn can acquire
    and release the same guard immediately. The warm's publish still runs after.
    """
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _seed_conversation(
            runtime, user_id="usr_warm", conversation_id="cnv_warm"
        )
        service = ContextCacheService(runtime)

        resolve_in_flight = asyncio.Event()
        foreground_acquired_guard = asyncio.Event()
        publish_called = asyncio.Event()

        async def slow_resolve(self, connection, **kwargs):
            resolve_in_flight.set()
            # Block until the foreground proves it acquired the guard while the
            # resolve is still running (i.e. the warm is NOT holding it).
            await asyncio.wait_for(foreground_acquired_guard.wait(), timeout=5.0)
            return SimpleNamespace(
                cache_key="cache_key_warm",
                from_cache=False,
                composed_context=SimpleNamespace(selected_memory_ids=[]),
            )

        async def fake_publish(self, resolution, *, last_retrieval_message_seq):
            publish_called.set()
            return True

        # ContextCacheService is a slots dataclass, so patch on the class.
        monkeypatch.setattr(
            ContextCacheService, "resolve_with_connection", slow_resolve
        )
        monkeypatch.setattr(
            ContextCacheService, "publish_pending_cache_entry", fake_publish
        )

        warm_task = asyncio.create_task(
            service._run_smart_fast_warm(
                user_id="usr_warm",
                conversation_id="cnv_warm",
                message_text="hello",
                assistant_mode_id=None,
                operational_profile=None,
                operational_signals=None,
                ablation=None,
                prompt_authority_context=normalize_request_authority_context(
                    privacy_enforcement="enforce",
                    user_id="usr_warm",
                    purpose="context_cache_fast",
                ),
                last_retrieval_message_seq=1,
            )
        )

        await asyncio.wait_for(resolve_in_flight.wait(), timeout=5.0)
        # Guard must be free while resolve is mid-flight.
        async with service.user_cache_guard("usr_warm"):
            foreground_acquired_guard.set()
        await asyncio.wait_for(warm_task, timeout=5.0)
        assert publish_called.is_set()
    finally:
        await runtime.close()
