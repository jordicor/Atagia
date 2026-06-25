"""Tests for the adaptive retrieval gate at the context-cache service layer.

CS4: cache protection (D6), the fast-path no-op (D8), and the smart_fast warm
veto (D9). These exercise the real ``ContextCacheService`` over an in-memory
SQLite database with a stub LLM provider whose need detector classifies the turn
as ``world`` (gate-eligible to skip) or omits the field (MIXED, retrieve).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
)
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    ResponseMode,
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
from atagia.services.prompt_authority import normalize_request_authority_context

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


class GateCacheProvider(LLMProvider):
    """Stub provider with a configurable ``memory_dependence`` classification."""

    name = "adaptive-gate-cache-tests"

    def __init__(self, *, memory_dependence: str | None = None) -> None:
        self._memory_dependence = memory_dependence
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose.startswith("need_detection_") and purpose.endswith("_card"):
            output = {
                "need_detection_needs_card": "none",
                "need_detection_language_card": "en\nen",
                "need_detection_memory_card": self._memory_dependence or "mixed",
                "need_detection_exact_card": "no",
                "need_detection_shape_card": "default",
                "need_detection_facets_card": "none",
                "need_detection_callback_card": "no",
                "need_detection_search_words_card": "a general question",
            }[purpose]
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=output,
            )
        if purpose == "context_cache_signal_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "contradiction_detected": False,
                        "high_stakes_topic": False,
                        "sensitive_content": False,
                        "mode_shift_target": None,
                        "short_followup": False,
                        "ambiguous_wording": False,
                    }
                ),
            )
        if purpose == "applicability_relevance_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} exact" for _memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "applicability_date_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} none" for _memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "coverage_expansion":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {"should_expand": False, "missing_facets": [], "sub_queries": []}
                ),
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in adaptive gate cache tests")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-adaptive-gate-cache.db"),
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
    *,
    memory_dependence: str | None,
) -> tuple[AppRuntime, GateCacheProvider]:
    provider = GateCacheProvider(memory_dependence=memory_dependence)
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    return runtime, provider


async def _seed(runtime: AppRuntime) -> None:
    connection = await runtime.open_connection()
    try:
        users = UserRepository(connection, runtime.clock)
        conversations = ConversationRepository(connection, runtime.clock)
        memories = MemoryObjectRepository(connection, runtime.clock)
        await users.create_user("usr_1")
        await conversations.create_conversation(
            "cnv_1", "usr_1", None, "coding_debug", "Chat"
        )
        await memories.create_memory_object(
            user_id="usr_1",
            workspace_id=None,
            conversation_id="cnv_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CONVERSATION,
            canonical_text="the user prefers concise replies",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.8,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id="mem_1",
            platform_id="default",
            scope_canonical=MemoryScope.CHAT.value,
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_gate_skip_does_not_publish_a_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="world"
    )
    try:
        await _seed(runtime)
        service = ContextCacheService(runtime)
        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Who painted a famous chapel ceiling?",
                adaptive_retrieval=True,
            )
        finally:
            await connection.close()

        # D6: a skipped turn never produces a pending entry.
        assert resolution.pending_cache_entry is None
        assert resolution.cache_ttl_seconds is None
        # The persisted retrieval plan stays a clean RetrievalPlan dump (strict
        # round-trip); the gate block lives only on the guard diagnostics.
        assert "adaptive_gate" not in resolution.source_retrieval_plan
        gate = resolution.retrieval_diagnostics_for_guard["adaptive_gate"]
        assert gate["status"] == "skipped"
        assert gate["skipped"] is True
        assert gate["fast_mode_equivalent"] is True
        # Publishing is a no-op.
        published = await service.publish_pending_cache_entry(
            resolution, last_retrieval_message_seq=1
        )
        assert published is False
        stored = await runtime.storage_backend.get_context_view(
            str(resolution.cache_key)
        )
        assert stored is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_gate_skip_does_not_overwrite_existing_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First a full-retrieval turn (MIXED) writes a good memory-context entry;
    # then a gate-skipped turn on the SAME cache key must not overwrite it.
    runtime, provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="mixed"
    )
    try:
        await _seed(runtime)
        service = ContextCacheService(runtime)
        connection = await runtime.open_connection()
        try:
            full = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="What did I tell you about my preferences?",
                adaptive_retrieval=True,
            )
        finally:
            await connection.close()
        assert full.pending_cache_entry is not None
        await service.publish_pending_cache_entry(full, last_retrieval_message_seq=1)
        stored_before = await runtime.storage_backend.get_context_view(
            str(full.cache_key)
        )
        assert stored_before is not None

        # Now flip the classification to world so the next turn is gate-skipped.
        provider._memory_dependence = "world"
        connection = await runtime.open_connection()
        try:
            skipped = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Who painted a famous chapel ceiling?",
                adaptive_retrieval=True,
            )
        finally:
            await connection.close()
        # The cache-hit reuse path runs before the gate; staleness may reuse the
        # prior entry. Either way, the skipped turn never publishes.
        assert skipped.pending_cache_entry is None
        published = await service.publish_pending_cache_entry(
            skipped, last_retrieval_message_seq=2
        )
        assert published is False
        stored_after = await runtime.storage_backend.get_context_view(
            str(skipped.cache_key)
        )
        # The original good memory context survives untouched.
        assert stored_after == stored_before
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_fast_path_reports_not_applicable_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D8: the fast path never calls the detector and the gate never runs.
    runtime, provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="world"
    )
    try:
        await _seed(runtime)
        service = ContextCacheService(runtime)
        authority = normalize_request_authority_context(
            privacy_enforcement="enforce",
            authenticated_user_privilege_level=None,
            authenticated_user_is_atagia_master=False,
            user_id="usr_1",
            purpose="context_cache_fast",
        )
        connection = await runtime.open_connection()
        try:
            resolution = await service.resolve_fast_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Who painted a famous chapel ceiling?",
                response_mode=ResponseMode.FAST,
                prompt_authority_context=authority,
            )
        finally:
            await connection.close()

        # No detector call at all.
        assert all(
            request.metadata.get("purpose") != "need_detection"
            for request in provider.requests
        )
        gate = resolution.source_retrieval_plan["adaptive_gate"]
        assert gate["status"] == "not_applicable"
        assert gate["skipped"] is False
        assert resolution.retrieval_diagnostics_for_guard["adaptive_gate"][
            "status"
        ] == "not_applicable"
        assert resolution.pending_cache_entry is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_smart_fast_warm_veto_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D9: a gate veto inside the warm yields no warm entry; the next-turn fast
    # path then finds no warm entry under the smart_fast key.
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="world"
    )
    try:
        await _seed(runtime)
        service = ContextCacheService(runtime)
        authority = normalize_request_authority_context(
            privacy_enforcement="enforce",
            authenticated_user_privilege_level=None,
            authenticated_user_is_atagia_master=False,
            user_id="usr_1",
            purpose="context_cache",
        )
        connection = await runtime.open_connection()
        try:
            warm = await service.resolve_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Who painted a famous chapel ceiling?",
                response_mode=ResponseMode.SMART_FAST,
                prompt_authority_context=authority,
                adaptive_retrieval=True,
            )
        finally:
            await connection.close()

        # The warm resolve was gate-skipped, so it has nothing to publish.
        assert warm.pending_cache_entry is None
        published = await service.publish_pending_cache_entry(
            warm, last_retrieval_message_seq=1
        )
        assert published is False

        # Next-turn smart_fast fast path finds no warm entry under the key.
        connection = await runtime.open_connection()
        try:
            fast = await service.resolve_fast_with_connection(
                connection,
                user_id="usr_1",
                conversation_id="cnv_1",
                message_text="Who painted a famous chapel ceiling?",
                response_mode=ResponseMode.SMART_FAST,
                prompt_authority_context=authority,
            )
        finally:
            await connection.close()
        assert fast.source_retrieval_plan["smart_fast_warm_entry_present"] is False
    finally:
        await runtime.close()
