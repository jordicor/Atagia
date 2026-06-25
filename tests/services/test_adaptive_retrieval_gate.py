"""Tests for the adaptive retrieval gate (pipeline short-circuit + threading).

CS3: pipeline gate behavior. The gate is one more output field of the existing
need-detector structured call, so these tests drive the pipeline directly with a
stub LLM provider that returns a chosen ``memory_dependence`` classification and
assert the short-circuit, the shadow-mode recording, and the conservative
"uncertainty -> retrieve" semantics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import re
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.memory.policy_manifest import (
    ManifestLoader,
    PolicyResolver,
    sync_assistant_modes,
)
from atagia.models.schemas_memory import (
    AdaptiveGateStatus,
    ExtractionConversationContext,
    MemoryDependence,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    RetrievalTrace,
)
from atagia.models.schemas_replay import AblationConfig
from atagia.services.embeddings import NoneBackend
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)
from atagia.services.retrieval_pipeline import RetrievalPipeline

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
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


class GateProvider(LLMProvider):
    """Stub provider returning a chosen ``memory_dependence`` classification."""

    name = "adaptive-gate-tests"

    def __init__(
        self,
        *,
        memory_dependence: str | None = None,
        exact_recall_needed: bool = False,
        sub_queries: list[str] | None = None,
        score_map: dict[str, float] | None = None,
        fail_purpose: str | None = None,
    ) -> None:
        self._memory_dependence = memory_dependence
        self._exact_recall_needed = exact_recall_needed
        self._sub_queries = sub_queries or ["general knowledge question"]
        self._score_map = dict(score_map or {})
        self._fail_purpose = fail_purpose
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata.get("purpose"))
        if self._fail_purpose is not None and (
            purpose == self._fail_purpose
            or (
                self._fail_purpose == "need_detection"
                and purpose.startswith("need_detection_")
                and purpose.endswith("_card")
            )
        ):
            self.requests.append(request)
            raise LLMError(f"Injected {self._fail_purpose} failure")
        self.requests.append(request)
        if purpose.startswith("need_detection_") and purpose.endswith("_card"):
            output = {
                "need_detection_needs_card": "none",
                "need_detection_language_card": "en\nen",
                "need_detection_memory_card": self._memory_dependence or "mixed",
                "need_detection_exact_card": (
                    "yes" if self._exact_recall_needed else "no"
                ),
                "need_detection_shape_card": "default",
                "need_detection_facets_card": "none",
                "need_detection_callback_card": "no",
                "need_detection_search_words_card": "\n".join(self._sub_queries[:6]) or "none",
            }[purpose]
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=output,
            )
        if purpose == "applicability_relevance_card":
            scores = [
                f"{score_key} {_label_for_score(self._score_map.get(memory_id, 0.5))}"
                for memory_id, score_key in _score_keys(request.messages[1].content)
            ]
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(scores),
            )
        if purpose == "applicability_date_card":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} none"
                    for _memory_id, score_key in _score_keys(request.messages[1].content)
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
        raise AssertionError("Embeddings are not used in adaptive gate tests")

    def purposes_called(self) -> list[str]:
        purposes = [str(request.metadata.get("purpose")) for request in self.requests]
        if any(
            purpose.startswith("need_detection_") and purpose.endswith("_card")
            for purpose in purposes
        ):
            purposes.append("need_detection")
        if "applicability_relevance_card" in purposes:
            purposes.append("applicability_scoring")
        return purposes


def _score_keys(prompt: str) -> list[tuple[str, str]]:
    return _CANDIDATE_SCORE_KEY_PATTERN.findall(prompt)


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
        small_corpus_token_threshold_ratio=0.0,
    )


async def _build_runtime(
    *,
    provider: GateProvider,
    mode_id: str = "coding_debug",
):
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(
        connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock
    )
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    await workspaces.create_workspace("wrk_1", "usr_1", "Workspace")
    await conversations.create_conversation("cnv_1", "usr_1", "wrk_1", mode_id, "Chat")
    pipeline = RetrievalPipeline(
        connection=connection,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        embedding_index=NoneBackend(),
        clock=clock,
        settings=_settings(),
    )
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()[mode_id]
    resolved_policy = PolicyResolver().resolve(manifest, None, None)
    context = ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id="wrk_1",
        assistant_mode_id=mode_id,
        recent_messages=[{"role": "user", "content": "earlier context"}],
    )
    return connection, memories, pipeline, resolved_policy, context


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    canonical_text: str,
) -> None:
    await memories.create_memory_object(
        user_id="usr_1",
        workspace_id="wrk_1",
        conversation_id="cnv_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.CONVERSATION,
        canonical_text=canonical_text,
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.8,
        privacy_level=0,
        status=MemoryStatus.ACTIVE,
        memory_id=memory_id,
        platform_id="default",
        scope_canonical=MemoryScope.CHAT.value,
    )


def _trace() -> RetrievalTrace:
    return RetrievalTrace(
        query_text="message",
        user_id="usr_1",
        conversation_id="cnv_1",
        timestamp_iso="2026-04-05T12:00:00+00:00",
    )


def _conversation_messages(message_text: str) -> list[dict[str, object]]:
    # Two-message transcript so the empty-clean and small-corpus shortcuts (which
    # require a single first user turn) do not pre-empt the gate decision point.
    return [
        {"id": "msg_0", "role": "user", "seq": 1, "text": "earlier context"},
        {"id": "msg_1", "role": "user", "seq": 2, "text": message_text},
    ]


@pytest.mark.asyncio
async def test_gate_on_world_skips_expensive_retrieval_stages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    message_text = "Who painted the ceiling of a famous chapel?"
    # The sub-query's tokens overlap the seeded memory so the base FTS lane
    # actually surfaces a candidate; the world-skip path then discards it,
    # exercising the discarded-base-candidate telemetry the gate logs.
    provider = GateProvider(
        memory_dependence="world",
        sub_queries=["famous chapel ceiling"],
        score_map={"mem_1": 0.9},
    )
    connection, memories, pipeline, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="The famous chapel ceiling restoration finished last year.",
        )
        trace = _trace()
        with caplog.at_level(
            logging.INFO, logger="atagia.services.retrieval_pipeline"
        ):
            result = await pipeline.execute(
                message_text=message_text,
                conversation_context=context,
                resolved_policy=resolved_policy,
                cold_start=False,
                conversation_messages=_conversation_messages(message_text),
                trace=trace,
                adaptive_retrieval=True,
            )

        # Skipped: status + classification recorded, no candidates composed.
        assert result.adaptive_gate_status is AdaptiveGateStatus.SKIPPED
        assert result.adaptive_gate_classification is MemoryDependence.WORLD
        assert result.raw_candidates == []
        assert result.scored_candidates == []
        assert result.composed_context.selected_memory_ids == []
        assert result.degraded_mode is False
        assert result.small_corpus_mode is False

        # The enriched/scoring/composition stages that would have run did not:
        # need detection ran (base candidates were searched then discarded), but
        # no applicability scoring LLM call was made.
        purposes = provider.purposes_called()
        assert "need_detection" in purposes
        assert "applicability_scoring" not in purposes

        # Skipped stages report zero timing.
        assert result.stage_timings["enriched_candidate_search"] == 0.0
        assert result.stage_timings["applicability_scoring"] == 0.0
        assert result.stage_timings["state_lookup"] == 0.0
        assert result.stage_timings["workspace_rollup_lookup"] == 0.0

        # Real need-detection trace survives.
        assert trace.need_detection is not None
        assert trace.need_detection.degraded_mode is False

        # The gate emits exactly one INFO line documenting the skip with the
        # world classification and the discarded-base-candidate count (the
        # seeded ``mem_1`` was searched, then dropped rather than composed).
        skip_records = [
            record
            for record in caplog.records
            if record.message.startswith("adaptive_gate_skip ")
        ]
        assert len(skip_records) == 1
        skip_record = skip_records[0]
        assert skip_record.levelno == logging.INFO
        assert skip_record.args[2] == MemoryDependence.WORLD.value
        assert skip_record.args[3] >= 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_gate_on_conversation_skips_but_keeps_contract() -> None:
    message_text = "Summarize what we just discussed."
    provider = GateProvider(memory_dependence="conversation")
    connection, memories, pipeline, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories, memory_id="mem_1", canonical_text="stored fact"
        )
        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=_conversation_messages(message_text),
            trace=_trace(),
            adaptive_retrieval=True,
        )

        assert result.adaptive_gate_status is AdaptiveGateStatus.SKIPPED
        assert result.adaptive_gate_classification is MemoryDependence.CONVERSATION
        assert result.scored_candidates == []
        # The contract lookup still ran (contract-only path).
        assert "contract_lookup" in result.stage_timings
        assert "applicability_scoring" not in provider.purposes_called()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_gate_on_conversation_with_exact_recall_runs_full_retrieval() -> None:
    message_text = "Did I mention the retry loop?"
    provider = GateProvider(
        memory_dependence="conversation",
        exact_recall_needed=True,
        sub_queries=["retry loop"],
        score_map={"mem_1": 0.9},
    )
    connection, memories, pipeline, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories,
            memory_id="mem_1",
            canonical_text="The user mentioned a retry loop during debugging.",
        )
        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=_conversation_messages(message_text),
            trace=_trace(),
            adaptive_retrieval=True,
        )

        assert result.adaptive_gate_status is AdaptiveGateStatus.RETRIEVED
        assert result.adaptive_gate_classification is MemoryDependence.CONVERSATION
        assert "applicability_scoring" in provider.purposes_called()
        assert result.scored_candidates
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_gate_on_personal_runs_full_retrieval() -> None:
    message_text = "What did I say about my retry loop last week?"
    provider = GateProvider(
        memory_dependence="personal",
        sub_queries=["retry loop"],
        score_map={"mem_1": 0.9},
    )
    connection, memories, pipeline, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories, memory_id="mem_1", canonical_text="retry loop discussion"
        )
        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=_conversation_messages(message_text),
            trace=_trace(),
            adaptive_retrieval=True,
        )

        # Flag on, but a personal turn retrieves: status RETRIEVED, full flow ran.
        assert result.adaptive_gate_status is AdaptiveGateStatus.RETRIEVED
        assert result.adaptive_gate_classification is MemoryDependence.PERSONAL
        assert "applicability_scoring" in provider.purposes_called()
        assert result.scored_candidates  # candidates were scored
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_gate_on_mixed_runs_full_retrieval() -> None:
    message_text = "How does the retry loop relate to what I told you before?"
    provider = GateProvider(
        memory_dependence="mixed",
        sub_queries=["retry loop"],
        score_map={"mem_1": 0.9},
    )
    connection, memories, pipeline, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories, memory_id="mem_1", canonical_text="retry loop discussion"
        )
        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=_conversation_messages(message_text),
            trace=_trace(),
            adaptive_retrieval=True,
        )

        assert result.adaptive_gate_status is AdaptiveGateStatus.RETRIEVED
        assert result.adaptive_gate_classification is MemoryDependence.MIXED
        # Applicability scoring runs on the full path; it never runs on a skip.
        assert "applicability_scoring" in provider.purposes_called()
        assert result.scored_candidates
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_gate_omission_defaults_to_mixed_and_retrieves() -> None:
    # The model omits memory_dependence entirely: it defaults to MIXED
    # (uncertainty -> retrieve), so even with the flag on the turn retrieves.
    message_text = "Tell me about the retry loop."
    provider = GateProvider(
        memory_dependence=None,
        sub_queries=["retry loop"],
        score_map={"mem_1": 0.9},
    )
    connection, memories, pipeline, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories, memory_id="mem_1", canonical_text="retry loop discussion"
        )
        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=_conversation_messages(message_text),
            trace=_trace(),
            adaptive_retrieval=True,
        )

        assert result.adaptive_gate_classification is MemoryDependence.MIXED
        assert result.adaptive_gate_status is AdaptiveGateStatus.RETRIEVED
        assert "applicability_scoring" in provider.purposes_called()
        assert result.scored_candidates
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_gate_off_records_shadow_status_without_changing_behavior() -> None:
    # Flag OFF: even a world classification retrieves fully, and the gate
    # records the classification under shadow status.
    message_text = "Tell me about the retry loop."
    provider = GateProvider(
        memory_dependence="world",
        sub_queries=["retry loop"],
        score_map={"mem_1": 0.9},
    )
    connection, memories, pipeline, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(
            memories, memory_id="mem_1", canonical_text="retry loop discussion"
        )
        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=_conversation_messages(message_text),
            trace=_trace(),
            adaptive_retrieval=False,
        )

        assert result.adaptive_gate_status is AdaptiveGateStatus.OFF_SHADOW
        assert result.adaptive_gate_classification is MemoryDependence.WORLD
        assert "applicability_scoring" in provider.purposes_called()
        assert result.scored_candidates
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_gate_has_no_authority_in_degraded_mode() -> None:
    # Need detection fails -> degraded mode. Even with the flag on and a world-
    # shaped query, the gate has no classification authority and the base search
    # behavior is unchanged (full path, RETRIEVED status, MIXED default class).
    message_text = "Who painted the ceiling of a famous chapel?"
    provider = GateProvider(
        memory_dependence="world",
        fail_purpose="need_detection",
        score_map={"mem_1": 0.9},
    )
    connection, memories, pipeline, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(memories, memory_id="mem_1", canonical_text="some fact")
        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=_conversation_messages(message_text),
            trace=_trace(),
            adaptive_retrieval=True,
        )

        assert result.degraded_mode is True
        assert result.adaptive_gate_status is AdaptiveGateStatus.RETRIEVED
        # Degraded path keeps the default MIXED classification (no detection).
        assert result.adaptive_gate_classification is MemoryDependence.MIXED
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_skip_need_detection_ablation_gives_gate_no_authority_and_retrieves() -> None:
    # With need detection ablated, the detector never runs, so the query
    # intelligence stays at the conservative MIXED default. The gate only fires
    # on world/conversation, so it has no authority here: the full path runs and
    # the result reports RETRIEVED (the flag is on).
    message_text = "Who painted the ceiling of a famous chapel?"
    provider = GateProvider(memory_dependence="world", score_map={"mem_1": 0.9})
    connection, memories, pipeline, resolved_policy, context = await _build_runtime(
        provider=provider
    )
    try:
        await _seed_memory(memories, memory_id="mem_1", canonical_text="some fact")
        result = await pipeline.execute(
            message_text=message_text,
            conversation_context=context,
            resolved_policy=resolved_policy,
            cold_start=False,
            conversation_messages=_conversation_messages(message_text),
            trace=_trace(),
            ablation=AblationConfig(skip_need_detection=True),
            adaptive_retrieval=True,
        )

        assert "need_detection" not in provider.purposes_called()
        # The default query intelligence is MIXED, so the gate (which only fires
        # on world/conversation) has no authority: full path, RETRIEVED status.
        assert result.adaptive_gate_status is AdaptiveGateStatus.RETRIEVED
        assert result.adaptive_gate_classification is MemoryDependence.MIXED
    finally:
        await connection.close()
