"""Multilingual retrieval pipeline tests."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, UserRepository, WorkspaceRepository
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver, sync_assistant_modes
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    RetrievalTrace,
    SpaceBoundaryMode,
)
from atagia.services.embeddings import NoneBackend
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.retrieval_pipeline import RetrievalPipeline

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


def _label_for_score(score: object) -> str:
    value = float(score)
    if value <= 0.10:
        return "drop"
    if value <= 0.40:
        return "weak"
    if value <= 0.65:
        return "useful"
    if value <= 0.85:
        return "strong"
    return "exact"


class MultilingualPipelineProvider(LLMProvider):
    name = "multilingual-pipeline-tests"

    def __init__(
        self,
        *,
        need_response: dict[str, object] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        self.need_response = need_response or {
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["¿Cuál es la dosis actual de amlodipino de Rosa?"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "¿Cuál es la dosis actual de amlodipino de Rosa?",
                    "fts_phrase": "amlodipino",
                    "must_keep_terms": ["Rosa", "amlodipino"],
                },
            ],
            "query_language": "es",
            "answer_language": "es",
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["medication"],
        }
        self.score_map = dict(score_map or {})
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose.startswith("need_detection_") and purpose.endswith("_card"):
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=self._need_card_output(purpose),
            )
        if purpose == "applicability_relevance_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(request.messages[1].content)
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} {_label_for_score(self.score_map.get(memory_id, 0.5))}"
                    for memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "applicability_date_card":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} none"
                    for _memory_id, score_key in _CANDIDATE_SCORE_KEY_PATTERN.findall(
                        request.messages[1].content
                    )
                ),
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    def _need_card_output(self, purpose: str) -> str:
        if purpose == "need_detection_needs_card":
            needs = self.need_response.get("needs")
            if not isinstance(needs, list) or not needs:
                return "none"
            labels = [
                str(item.get("need_type"))
                for item in needs
                if isinstance(item, dict) and item.get("need_type")
            ]
            return "\n".join(labels) if labels else "none"
        if purpose == "need_detection_language_card":
            return "\n".join(
                [
                    str(self.need_response.get("query_language") or "en"),
                    str(self.need_response.get("answer_language") or "en"),
                ]
            )
        if purpose == "need_detection_memory_card":
            return str(self.need_response.get("memory_dependence") or "mixed")
        if purpose == "need_detection_exact_card":
            return "yes" if self.need_response.get("exact_recall_needed") else "no"
        if purpose == "need_detection_shape_card":
            return {
                "slot_fill": "slot",
                "broad_list": "list",
                "temporal": "time",
                "default": "default",
            }.get(str(self.need_response.get("query_type") or "default"), "default")
        if purpose == "need_detection_facets_card":
            raw_facets = self.need_response.get("exact_facets")
            if not isinstance(raw_facets, list) or not raw_facets:
                return "none"
            mapping = {"other_verbatim": "wording"}
            return "\n".join(mapping.get(str(facet), str(facet)) for facet in raw_facets)
        if purpose == "need_detection_callback_card":
            return "yes" if self.need_response.get("callback_bias") else "no"
        if purpose == "need_detection_search_words_card":
            terms: list[str] = []
            first_hint = next(
                (
                    hint
                    for hint in self.need_response.get("sparse_query_hints") or []
                    if isinstance(hint, dict)
                ),
                None,
            )
            if first_hint is not None:
                for field in ("must_keep_terms", "quoted_phrases"):
                    values = first_hint.get(field)
                    if isinstance(values, list):
                        terms.extend(str(value) for value in values if str(value).strip())
                if not terms and first_hint.get("fts_phrase"):
                    terms.append(str(first_hint["fts_phrase"]))
            return "\n".join(terms[:6]) if terms else "none"
        if purpose == "need_detection_search_words_other_language_card":
            return "none"
        raise AssertionError(f"Unexpected card purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in retrieval pipeline tests")


def _settings(*, small_corpus_token_threshold_ratio: float = 0.0) -> Settings:
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
        small_corpus_token_threshold_ratio=small_corpus_token_threshold_ratio,
    )


async def _build_runtime(
    *,
    mode_id: str = "coding_debug",
    provider: MultilingualPipelineProvider | None = None,
    settings: Settings | None = None,
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock)
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    contracts = ContractDimensionRepository(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", mode_id, "Chat")
    llm_provider = provider or MultilingualPipelineProvider()
    resolved_settings = settings or _settings()
    pipeline = RetrievalPipeline(
        connection=connection,
        llm_client=LLMClient(provider_name=llm_provider.name, providers=[llm_provider]),
        embedding_index=NoneBackend(),
        clock=clock,
        settings=resolved_settings,
    )
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()[mode_id]
    resolved_policy = PolicyResolver().resolve(manifest, None, None)
    context = ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id="wrk_1",
        assistant_mode_id=mode_id,
        recent_messages=[],
    )
    return connection, memories, contracts, pipeline, llm_provider, resolved_policy, context


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    canonical_text: str,
    scope: MemoryScope,
    object_type: MemoryObjectType = MemoryObjectType.EVIDENCE,
    assistant_mode_id: str = "coding_debug",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    privacy_level: int = 0,
    language_codes: list[str] | None = None,
    space_id: str | None = None,
    space_boundary_mode: str | None = None,
) -> dict[str, object]:
    scope_canonical = {
        MemoryScope.CONVERSATION: MemoryScope.CHAT.value,
        MemoryScope.EPHEMERAL_SESSION: MemoryScope.CHAT.value,
        MemoryScope.WORKSPACE: MemoryScope.CHARACTER.value,
        MemoryScope.GLOBAL_USER: MemoryScope.USER.value,
    }.get(scope)
    return await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1" if scope is MemoryScope.CONVERSATION else None,
        assistant_mode_id=assistant_mode_id,
        object_type=object_type,
        scope=scope,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED if object_type is not MemoryObjectType.INTERACTION_CONTRACT else MemorySourceKind.INFERRED,
        confidence=0.8,
        privacy_level=privacy_level,
        status=status,
        language_codes=language_codes,
        memory_id=memory_id,
        platform_id="default",
        character_id="wrk_1" if scope is MemoryScope.WORKSPACE else None,
        scope_canonical=scope_canonical,
        space_id=space_id,
        space_boundary_mode=space_boundary_mode,
    )


def _score_request_memory_ids(provider: MultilingualPipelineProvider) -> list[str]:
    memory_ids: list[str] = []
    for request in provider.requests:
        if str(request.metadata.get("purpose")) != "applicability_relevance_card":
            continue
        memory_ids.extend(_MEMORY_ID_PATTERN.findall(request.messages[1].content))
    return memory_ids


def _language_card_prompt(provider: MultilingualPipelineProvider) -> str:
    return next(
        request.messages[1].content
        for request in provider.requests
        if str(request.metadata.get("purpose")) == "need_detection_language_card"
    )


def _saved_language_profile_block(prompt: str) -> str:
    return prompt.split("Saved memory languages:\n", 1)[1].split(
        "\nUser communication profile:",
        1,
    )[0]


@pytest.mark.asyncio
async def test_pipeline_uses_language_profile_with_parallel_cards_and_literal_anchors() -> None:
    message_text = "¿Cuál es la dosis actual de amlodipino de Rosa?"
    provider = MultilingualPipelineProvider(
        score_map={"mem_english": 0.94},
    )
    connection, memories, _contracts, pipeline, provider, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_english",
            canonical_text="Rosa toma amlodipino 10 mg los martes.",
            scope=MemoryScope.CONVERSATION,
            language_codes=["en"],
        )
        await _seed_memory(
            memories,
            memory_id="mem_pending_fr",
            canonical_text="dose actuelle d'amlodipine",
            scope=MemoryScope.CONVERSATION,
            status=MemoryStatus.PENDING_USER_CONFIRMATION,
            language_codes=["fr"],
        )
        await _seed_memory(
            memories,
            memory_id="mem_private_de",
            canonical_text="aktuelle amlodipin dosis",
            scope=MemoryScope.CONVERSATION,
            privacy_level=3,
            language_codes=["de"],
        )
        await _seed_memory(
            memories,
            memory_id="mem_missing_codes",
            canonical_text="memory without language metadata",
            scope=MemoryScope.CONVERSATION,
        )
        trace = RetrievalTrace(
            query_text=message_text,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            timestamp_iso="2026-04-05T12:00:00Z",
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
            trace=trace,
        )

        prompt = _language_card_prompt(provider)
        profile_block = _saved_language_profile_block(prompt)
        assert profile_block == "\n".join(
            [
                "en: 1 memories (last seen 2026-04-05)",
                "unknown: 1 memories (last seen 2026-04-05)",
            ]
        )
        assert "fr:" not in profile_block
        assert "de:" not in profile_block
        assert "amlodipine" not in profile_block
        assert [plan.text for plan in result.retrieval_plan.sub_query_plans] == [
            "¿Cuál es la dosis actual de amlodipino de Rosa?",
        ]
        assert "amlodipino" in " ".join(result.retrieval_plan.fts_queries)
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_english"]
        assert [candidate.memory_id for candidate in result.scored_candidates] == ["mem_english"]
        assert result.composed_context.selected_memory_ids == ["mem_english"]
        assert _score_request_memory_ids(provider) == ["mem_english"]
        assert trace.need_detection is not None
        assert trace.need_detection.query_language == "es"
        assert trace.need_detection.answer_language == "es"
        assert [
            row.model_dump(mode="json")
            for row in trace.need_detection.content_language_profile
        ] == [
            {
                "language_code": "en",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            },
            {
                "language_code": "unknown",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            },
        ]
        assert [anchor.anchor_type for anchor in trace.need_detection.anchors] == [
            "unknown",
            "unknown",
        ]
        assert trace.need_detection.anchors[0].preserve_verbatim is True
        assert [anchor.original_surface for anchor in trace.need_detection.anchors] == [
            "Rosa",
            "amlodipino",
        ]
        assert trace.need_detection.alias_groups == []
        assert trace.candidate_search is not None
        fts_execution_sources = [
            execution.source
            for counts in trace.candidate_search.per_subquery_counts
            for execution in counts.fts_query_executions
        ]
        assert "alias" not in fts_execution_sources
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_traces_unknown_only_language_profile_without_bridge_target() -> None:
    message_text = "Cual es la direccion del nuevo apartamento de Ben?"
    provider = MultilingualPipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["dirección del nuevo apartamento de Ben"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "dirección del nuevo apartamento de Ben",
                    "fts_phrase": "Ben apartamento dirección",
                    "must_keep_terms": ["Ben", "apartamento", "dirección"],
                }
            ],
            "query_language": "es",
            "answer_language": "es",
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["location"],
        },
    )
    connection, memories, _contracts, pipeline, provider, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_unknown_language",
            canonical_text="The lease was signed for 4217 Fremont Avenue North.",
            scope=MemoryScope.CONVERSATION,
        )
        trace = RetrievalTrace(
            query_text=message_text,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            timestamp_iso="2026-04-05T12:00:00Z",
        )

        await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
            trace=trace,
        )

        prompt = _language_card_prompt(provider)
        profile_block = _saved_language_profile_block(prompt)
        assert profile_block == "unknown: 1 memories (last seen 2026-04-05)"
        assert trace.need_detection is not None
        assert [
            row.model_dump(mode="json")
            for row in trace.need_detection.content_language_profile
        ] == [
            {
                "language_code": "unknown",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            }
        ]
        assert trace.need_detection.sub_queries == [message_text]
        assert trace.need_detection.alias_groups == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_phase8_need_detection_trace_language_profile_is_content_free_and_alias_independent() -> None:
    message_text = "¿Dónde está el documento de alquiler?"
    provider = MultilingualPipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["documento alquiler"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "documento alquiler",
                    "fts_phrase": "documento alquiler",
                }
            ],
            "query_language": "es",
            "answer_language": "es",
            "query_type": "default",
            "retrieval_levels": [0],
            "exact_recall_needed": False,
            "exact_facets": [],
        },
    )
    connection, memories, _contracts, pipeline, provider, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_profile_en",
            canonical_text="The canonical lease profile marker is in English.",
            scope=MemoryScope.CONVERSATION,
            language_codes=["en"],
        )
        trace = RetrievalTrace(
            query_text=message_text,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            timestamp_iso="2026-04-05T12:00:00Z",
        )

        await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
            trace=trace,
        )

        assert trace.need_detection is not None
        assert trace.need_detection.alias_groups == []
        profile_rows = [
            row.model_dump(mode="json")
            for row in trace.need_detection.content_language_profile
        ]
        assert profile_rows == [
            {
                "language_code": "en",
                "memory_count": 1,
                "last_seen_at": "2026-04-05T12:00:00+00:00",
            }
        ]
        assert all(
            set(row) == {"language_code", "memory_count", "last_seen_at"}
            for row in profile_rows
        )
        profile_json = json.dumps(profile_rows, sort_keys=True)
        for forbidden in [
            "mem_profile_en",
            "canonical",
            "lease profile marker",
            "documento alquiler",
        ]:
            assert forbidden not in profile_json
        assert '"de"' not in profile_json
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_literal_anchor_lane_recovers_without_alias_evidence() -> None:
    message_text = "¿Cuál es la dosis actual de amlodipino?"
    provider = MultilingualPipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": [message_text],
            "sparse_query_hints": [
                {
                    "sub_query_text": message_text,
                    "fts_phrase": "dosis amlodipino",
                    "must_keep_terms": ["dosis", "amlodipino"],
                }
            ],
            "query_language": "es",
            "answer_language": "es",
            "query_type": "slot_fill",
            "retrieval_levels": [0],
            "exact_recall_needed": True,
            "exact_facets": ["medication"],
        },
        score_map={"mem_english": 0.94},
    )
    connection, memories, _contracts, pipeline, provider, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_english",
            canonical_text="Rosa toma amlodipino 10 mg los martes.",
            scope=MemoryScope.CONVERSATION,
            language_codes=["en"],
        )
        trace = RetrievalTrace(
            query_text=message_text,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            timestamp_iso="2026-04-05T12:00:00Z",
        )

        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
            trace=trace,
        )

        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_english"]
        assert result.composed_context.selected_memory_ids == ["mem_english"]
        assert "Rosa toma amlodipino 10 mg los martes." in result.composed_context.memory_block
        assert "alias_anchor" not in result.composed_context.memory_block
        assert "runtime_alias_or" not in result.composed_context.memory_block
        assert trace.candidate_search is not None
        executions = [
            execution
            for counts in trace.candidate_search.per_subquery_counts
            for execution in counts.fts_query_executions
        ]
        assert not any(execution.source == "alias_anchor" for execution in executions)
        assert not any(execution.kind == "corpus_near_or" for execution in executions)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_language_profile_respects_active_space_context() -> None:
    message_text = "¿Qué recuerdas de este espacio?"
    provider = MultilingualPipelineProvider()
    connection, memories, _contracts, pipeline, provider, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    context = context.model_copy(
        update={
            "active_space_id": "space_active",
            "active_space_boundary_mode": SpaceBoundaryMode.SEVERANCE,
        }
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_active_space_en",
            canonical_text="active space english memory",
            scope=MemoryScope.CONVERSATION,
            language_codes=["en"],
            space_id="space_active",
            space_boundary_mode=SpaceBoundaryMode.FOCUS.value,
        )
        await _seed_memory(
            memories,
            memory_id="mem_other_space_es",
            canonical_text="other space spanish memory",
            scope=MemoryScope.CONVERSATION,
            language_codes=["es"],
            space_id="space_other",
            space_boundary_mode=SpaceBoundaryMode.FOCUS.value,
        )

        await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=[{"role": "user", "text": message_text}],
        )

        prompt = _language_card_prompt(provider)
        profile_block = _saved_language_profile_block(prompt)
        assert profile_block == "en: 1 memories (last seen 2026-04-05)"
        assert "es:" not in profile_block
    finally:
        await connection.close()
