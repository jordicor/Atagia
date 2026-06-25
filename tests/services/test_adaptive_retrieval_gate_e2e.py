"""End-to-end adaptive retrieval gate tests over the full chat flow.

CS5: drive the real ``ChatService.chat_reply`` and ``SidecarService.get_context``
over an in-memory SQLite database (never mocked) with only the LLM stubbed, and
assert the full six-cell semantics table from the plan:

| response_mode | adaptive_retrieval | Behavior                                  |
|---------------|--------------------|-------------------------------------------|
| fast          | OFF or ON          | Identical; gate ``not_applicable``        |
| normal        | OFF                | Full retrieval; gate shadow status        |
| normal        | ON / world,conv    | Skip; contract-only; no cache publish     |
| normal        | ON / personal,mixed| Full retrieval                            |
| smart_fast    | OFF                | Warm publishes                            |
| smart_fast    | ON / world         | Warm vetoed; next turn finds no warm entry|

Plus the invariants: a gate-skipped turn never overwrites an existing cache
entry, ingest/extraction is still enqueued on a gate-skipped turn, and a
degraded need-detection leaves the gate without authority.
"""

from __future__ import annotations

import asyncio
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
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    ResponseMode,
)
from atagia.services.chat_service import ChatService
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
    LLMProvider,
)
from atagia.services.sidecar_service import SidecarService

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


def _date_card_output(prompt: str) -> str:
    return "\n".join(
        f"{score_key} none"
        for _memory_id, score_key in _CANDIDATE_SCORE_KEY_PATTERN.findall(prompt)
    )


class GateChatProvider(LLMProvider):
    """Full-chat stub provider with a configurable ``memory_dependence``.

    Handles every purpose a complete chat turn can request: need detection
    (and its optional review/coverage sub-purposes), the context-cache signal
    detection, applicability scoring, the chat reply, and the answer
    postcondition verification. ``memory_dependence`` is mutable so a single
    runtime can exercise a full-retrieval turn followed by a gate-skipped turn.
    """

    name = "adaptive-gate-e2e-tests"

    def __init__(
        self,
        *,
        memory_dependence: str | None = None,
        fail_purpose: str | None = None,
    ) -> None:
        self._memory_dependence = memory_dependence
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
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=_date_card_output(request.messages[1].content),
            )
        if purpose == "coverage_expansion":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {"should_expand": False, "missing_facets": [], "sub_queries": []}
                ),
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Here is the answer.",
            )
        if purpose == "answer_postcondition_verification":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "readable": True,
                        "is_abstention": False,
                        "contains_concrete_claims": True,
                        "unsupported_concrete_claims": False,
                        "covers_requested_facets": True,
                        "requires_abstention": False,
                        "pass_postcondition": True,
                        "failure_reasons": [],
                        "explanation": "Supported for the test.",
                    }
                ),
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in adaptive gate e2e tests")

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

    def count_purpose(self, purpose: str) -> int:
        return sum(1 for called in self.purposes_called() if called == purpose)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = dict(
        sqlite_path=str(tmp_path / "atagia-adaptive-gate-e2e.db"),
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
    base.update(overrides)
    return Settings(**base)


async def _build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    memory_dependence: str | None,
    fail_purpose: str | None = None,
) -> tuple[AppRuntime, GateChatProvider]:
    provider = GateChatProvider(
        memory_dependence=memory_dependence, fail_purpose=fail_purpose
    )
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    return runtime, provider


async def _seed_conversation(
    runtime: AppRuntime,
    *,
    user_id: str = "usr_1",
    conversation_id: str = "cnv_1",
    assistant_mode_id: str = "coding_debug",
) -> None:
    connection = await runtime.open_connection()
    try:
        users = UserRepository(connection, runtime.clock)
        conversations = ConversationRepository(connection, runtime.clock)
        await users.create_user(user_id)
        await conversations.create_conversation(
            conversation_id, user_id, None, assistant_mode_id, "Chat"
        )
    finally:
        await connection.close()


async def _seed_user_memory(
    runtime: AppRuntime,
    *,
    user_id: str = "usr_1",
    memory_id: str,
    canonical_text: str,
) -> None:
    connection = await runtime.open_connection()
    try:
        await MemoryObjectRepository(connection, runtime.clock).create_memory_object(
            user_id=user_id,
            workspace_id=None,
            conversation_id=None,
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text=canonical_text,
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            status=MemoryStatus.ACTIVE,
            memory_id=memory_id,
            platform_id="default",
            scope_canonical=MemoryScope.USER.value,
        )
    finally:
        await connection.close()


async def _drain_background_tasks(runtime: AppRuntime) -> None:
    for _ in range(50):
        tasks = [task for task in runtime._background_tasks if not task.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)
    raise AssertionError("Background tasks did not settle")


def _profile_token(runtime: AppRuntime) -> str:
    from atagia.services.chat_support import default_operational_profile_snapshot

    return default_operational_profile_snapshot(
        loader=runtime.operational_profile_loader,
        settings=runtime.settings,
    ).token


async def _conversation_snapshot(
    runtime: AppRuntime,
    *,
    user_id: str = "usr_1",
    conversation_id: str = "cnv_1",
) -> dict[str, object]:
    connection = await runtime.open_connection()
    try:
        conversation = await ConversationRepository(
            connection, runtime.clock
        ).get_conversation(conversation_id, user_id)
        assert conversation is not None
        return conversation
    finally:
        await connection.close()


def _smart_fast_cache_key(
    runtime: AppRuntime,
    cache_service: ContextCacheService,
    snapshot: dict[str, object],
) -> str:
    return cache_service.build_cache_key(
        user_id="usr_1",
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        workspace_id=snapshot.get("workspace_id"),
        active_presence_id=snapshot.get("active_presence_id"),
        active_space_id=snapshot.get("active_space_id"),
        active_mind_id=snapshot.get("active_mind_id"),
        mind_topology=snapshot.get("mind_topology"),
        active_embodiment_id=snapshot.get("active_embodiment_id"),
        active_realm_id=snapshot.get("active_realm_id"),
        operational_profile_token=_profile_token(runtime),
        response_mode=ResponseMode.SMART_FAST,
    )


async def _stored_event(runtime: AppRuntime, retrieval_event_id: str) -> dict[str, object]:
    connection = await runtime.open_connection()
    try:
        event = await RetrievalEventRepository(connection, runtime.clock).get_event(
            retrieval_event_id, "usr_1"
        )
        assert event is not None
        return event
    finally:
        await connection.close()


async def _worker_job_count(runtime: AppRuntime) -> int:
    connection = await runtime.open_connection()
    try:
        cursor = await connection.execute(
            "SELECT COUNT(*) AS count FROM worker_job_runs"
        )
        row = await cursor.fetchone()
        return int(row["count"])
    finally:
        await connection.close()


# ---------------------------------------------------------------------------
# fast OFF / ON are identical (not_applicable diagnostics)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("adaptive_retrieval", [False, True])
async def test_fast_mode_is_identical_with_flag_off_or_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adaptive_retrieval: bool,
) -> None:
    # Fast mode never retrieves; the flag is a documented no-op and the gate is
    # reported ``not_applicable`` regardless of the flag value.
    runtime, provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="world"
    )
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Who painted a famous chapel ceiling?",
            assistant_mode_id="coding_debug",
            response_mode=ResponseMode.FAST,
            adaptive_retrieval=adaptive_retrieval,
            debug=True,
        )

        # The detector never runs on the fast path, flag on or off.
        assert "need_detection" not in provider.purposes_called()
        assert "applicability_scoring" not in provider.purposes_called()
        assert result.debug is not None
        assert result.debug["adaptive_retrieval"] is adaptive_retrieval
        gate = result.debug["retrieval_plan"]["adaptive_gate"]
        assert gate["status"] == "not_applicable"
        assert gate["skipped"] is False
        assert gate["fast_mode_equivalent"] is True
        guard = result.debug["retrieval_diagnostics_for_guard"]["adaptive_gate"]
        assert guard["status"] == "not_applicable"
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# normal OFF: full retrieval + shadow status recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_off_runs_full_retrieval_and_records_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag OFF: even a world classification retrieves fully (zero behavior
    # change) and the gate records the classification under the shadow status.
    runtime, provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="world"
    )
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Tell me about the budget.",
            assistant_mode_id="coding_debug",
            response_mode=ResponseMode.NORMAL,
            adaptive_retrieval=False,
            debug=True,
        )

        # Full retrieval ran despite the world classification.
        assert "need_detection" in provider.purposes_called()
        assert "applicability_scoring" in provider.purposes_called()
        assert result.debug is not None
        assert result.debug["adaptive_retrieval"] is False
        # The gate block lives on the guard diagnostics, never on the persisted
        # retrieval plan (which must stay a strict-round-trippable plan dump).
        assert "adaptive_gate" not in result.debug["retrieval_plan"]
        gate = result.debug["retrieval_diagnostics_for_guard"]["adaptive_gate"]
        assert gate["status"] == "shadow"
        assert gate["classification"] == "world"
        assert gate["skipped"] is False
        assert gate["fast_mode_equivalent"] is False

        # The shadow status and resolved flag are persisted on the event too: the
        # plan JSON stays clean, the gate rides on the guard diagnostics.
        event = await _stored_event(runtime, result.retrieval_event_id)
        assert event["outcome_json"]["adaptive_retrieval"] is False
        assert "adaptive_gate" not in event["retrieval_plan_json"]
        assert (
            event["outcome_json"]["retrieval_diagnostics_for_guard"]["adaptive_gate"][
                "status"
            ]
            == "shadow"
        )
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# normal ON: skip on world/conversation, full retrieval on personal/mixed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("classification", "expected_status", "expected_classification", "scores_expected"),
    [
        ("world", "skipped", "world", False),
        ("conversation", "retrieved", "mixed", False),
    ],
)
async def test_normal_on_handles_world_and_conversation_without_recent_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    classification: str,
    expected_status: str,
    expected_classification: str,
    scores_expected: bool,
) -> None:
    runtime, provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence=classification
    )
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Summarize what we just covered.",
            assistant_mode_id="coding_debug",
            response_mode=ResponseMode.NORMAL,
            adaptive_retrieval=True,
            debug=True,
        )

        assert "need_detection" in provider.purposes_called()
        if scores_expected:
            assert "applicability_scoring" in provider.purposes_called()
        else:
            assert "applicability_scoring" not in provider.purposes_called()
            assert result.composed_context.selected_memory_ids == []
        assert result.debug is not None
        gate = result.debug["retrieval_diagnostics_for_guard"]["adaptive_gate"]
        assert gate["status"] == expected_status
        assert gate["classification"] == expected_classification
        assert gate["skipped"] is (expected_status == "skipped")
        assert gate["fast_mode_equivalent"] is (expected_status == "skipped")
        assert result.response_text == "Here is the answer."
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("classification", ["personal", "mixed"])
async def test_normal_on_runs_full_retrieval_on_personal_and_mixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    classification: str,
) -> None:
    runtime, provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence=classification
    )
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What did I tell you about the budget?",
            assistant_mode_id="coding_debug",
            response_mode=ResponseMode.NORMAL,
            adaptive_retrieval=True,
            debug=True,
        )

        # A memory-dependent turn retrieves fully even with the flag on.
        assert "need_detection" in provider.purposes_called()
        assert "applicability_scoring" in provider.purposes_called()
        assert result.debug is not None
        assert "adaptive_gate" not in result.debug["retrieval_plan"]
        gate = result.debug["retrieval_diagnostics_for_guard"]["adaptive_gate"]
        assert gate["status"] == "retrieved"
        assert gate["classification"] == classification
        assert gate["skipped"] is False
        assert gate["fast_mode_equivalent"] is False
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# smart_fast OFF: warm publishes; ON / world: warm vetoed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smart_fast_off_warm_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag OFF: the background warm runs the full pipeline (world is shadow-only)
    # and publishes the warm entry; the next smart_fast turn folds in the
    # retrieved memory.
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="world"
    )
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )
        sidecar = SidecarService(runtime)
        cache_service = ContextCacheService(runtime)

        first = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="What is the apartment budget?",
            mode="coding_debug",
            response_mode="smart_fast",
            adaptive_retrieval=False,
            message_id="host-user-1",
        )
        assert first.response_mode == "smart_fast"
        await _drain_background_tasks(runtime)

        snapshot = await _conversation_snapshot(runtime)
        warm_key = _smart_fast_cache_key(runtime, cache_service, snapshot)
        warmed = await runtime.storage_backend.get_context_view(warm_key)
        assert warmed is not None, "flag-off warm must publish its keyspace"

        # The next smart_fast turn reads the warmed memory context.
        second = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Remind me of the budget.",
            mode="coding_debug",
            response_mode="smart_fast",
            adaptive_retrieval=False,
            message_id="host-user-2",
        )
        assert "mem_budget" in {m.memory_id for m in second.memories}
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_smart_fast_on_warm_vetoed_on_world_and_next_turn_finds_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flag ON: the background warm runs its pipeline, the gate vetoes it on the
    # world classification (publish=False), so no warm entry is written and the
    # next smart_fast turn finds nothing warmed.
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="world"
    )
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )
        sidecar = SidecarService(runtime)
        cache_service = ContextCacheService(runtime)

        first = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Who painted a famous chapel ceiling?",
            mode="coding_debug",
            response_mode="smart_fast",
            adaptive_retrieval=True,
            message_id="host-user-1",
        )
        assert first.response_mode == "smart_fast"
        await _drain_background_tasks(runtime)

        snapshot = await _conversation_snapshot(runtime)
        warm_key = _smart_fast_cache_key(runtime, cache_service, snapshot)
        # D9: the warm was gate-vetoed, so nothing was published.
        assert await runtime.storage_backend.get_context_view(warm_key) is None

        # The next smart_fast turn's fast path finds no warm entry.
        second = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Who painted a famous chapel ceiling?",
            mode="coding_debug",
            response_mode="smart_fast",
            adaptive_retrieval=True,
            message_id="host-user-2",
        )
        assert "mem_budget" not in {m.memory_id for m in second.memories}
        await _drain_background_tasks(runtime)
        assert await runtime.storage_backend.get_context_view(warm_key) is None
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# invariants: cache protection, ingest enqueue, degraded authority
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_skip_turn_never_overwrites_existing_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A full-retrieval turn (mixed) writes a good memory-context entry; a later
    # gate-skipped turn on the same cache key must not overwrite it (D6/D7).
    runtime, provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="mixed"
    )
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )
        chat = ChatService(runtime)

        full = await chat.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="What did I tell you about the budget?",
            assistant_mode_id="coding_debug",
            response_mode=ResponseMode.NORMAL,
            adaptive_retrieval=True,
            debug=True,
        )
        assert full.debug is not None
        full_gate = full.debug["retrieval_diagnostics_for_guard"]["adaptive_gate"]
        assert full_gate["status"] == "retrieved"
        cache_key = full.debug["cache"]["cache_key"]
        stored_before = await runtime.storage_backend.get_context_view(str(cache_key))
        assert stored_before is not None

        # Flip the classification to world: the next turn is gate-skipped.
        provider._memory_dependence = "world"
        skipped = await chat.chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Who painted a famous chapel ceiling?",
            assistant_mode_id="coding_debug",
            response_mode=ResponseMode.NORMAL,
            adaptive_retrieval=True,
            debug=True,
        )
        assert skipped.debug is not None
        assert (
            skipped.debug["retrieval_diagnostics_for_guard"]["adaptive_gate"]["status"]
            == "skipped"
        )

        stored_after = await runtime.storage_backend.get_context_view(str(cache_key))
        # The original good memory context survives untouched.
        assert stored_after == stored_before
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_gate_skip_turn_still_enqueues_ingest_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # D11: memory extraction/ingest runs for every turn regardless of the gate
    # outcome. A gate-skipped turn must still enqueue post-response jobs.
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="world"
    )
    try:
        await _seed_conversation(runtime)
        # Seed a memory so the turn is not a cold start: the cold-start
        # empty-clean shortcut runs BEFORE the gate (D4), so a seeded corpus is
        # what makes the turn actually reach and exercise the gate-skip path.
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )
        assert await _worker_job_count(runtime) == 0

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Who painted a famous chapel ceiling?",
            assistant_mode_id="coding_debug",
            response_mode=ResponseMode.NORMAL,
            adaptive_retrieval=True,
            debug=True,
        )
        assert result.debug is not None
        assert (
            result.debug["retrieval_diagnostics_for_guard"]["adaptive_gate"]["status"]
            == "skipped"
        )
        # Post-response jobs were enqueued despite the gate skip.
        assert result.debug["enqueued_job_ids"]
        assert await _worker_job_count(runtime) > 0
        event = await _stored_event(runtime, result.retrieval_event_id)
        assert event["outcome_json"]["background_tasks_enqueued"] is True
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_degraded_need_detection_leaves_gate_without_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Need detection fails -> degraded mode. Even with the flag on and a world-
    # shaped query, the gate has no authority: full retrieval runs and the gate
    # keeps the conservative MIXED default with a RETRIEVED status.
    runtime, provider = await _build_runtime(
        tmp_path,
        monkeypatch,
        memory_dependence="world",
        fail_purpose="need_detection",
    )
    try:
        await _seed_conversation(runtime)
        await _seed_user_memory(
            runtime,
            memory_id="mem_budget",
            canonical_text="The apartment budget is 2800 dollars.",
        )

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Who painted a famous chapel ceiling?",
            assistant_mode_id="coding_debug",
            response_mode=ResponseMode.NORMAL,
            adaptive_retrieval=True,
            debug=True,
        )

        # The detector was attempted and failed; the base-search path stayed in
        # control (degraded). The gate never skipped.
        assert "need_detection" in provider.purposes_called()
        assert result.debug is not None
        gate = result.debug["retrieval_diagnostics_for_guard"]["adaptive_gate"]
        assert gate["status"] == "retrieved"
        assert gate["classification"] == "mixed"
        assert gate["skipped"] is False
    finally:
        await runtime.close()
