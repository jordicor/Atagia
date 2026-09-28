"""Single-turn interactive latency probe for the LoCoMo harness.

Every historical LoCoMo latency number was measured under contention (multiple
questions scored in parallel across conversations) and against per-question cold
runtimes (``_score_question_on_db_snapshot`` copies the retained DB and boots a
fresh ``Atagia`` engine for every question). There is therefore no honest
measurement of real interactive single-turn latency.

This probe fills that gap. It runs exactly ONE question, with NO parallelism
(concurrency is hard-set to 1), against a retained benchmark DB reused through
the same ``reuse_db`` / ``reuse_db_dir`` machinery evaluate-only mode uses. It
measures the wall time of the synchronous ``engine.chat`` call plus the
per-stage timings the retrieval trace already emits, and labels the cache/warm
state of each measurement explicitly.

Cold vs warm is derived from the engine's real lifecycle, not guessed from the
loop index: a SINGLE ``Atagia`` engine is constructed against a fresh DB
snapshot, and the probe records how many turns that engine had already served
before each call. ``repeat 0`` is the first turn served by this
freshly-constructed engine instance (engine-level cold: fresh connection pool
and engine state; the label says nothing about process-global warmth, though
no process-global caches exist in src/atagia today); ``repeat >0`` reuse the
same warm engine. The ``turns_served_before_call`` field makes that
derivation inspectable.

Engine age is not cache state, so each row also carries the context-cache state
the engine itself reported for that turn (``state.cache``, read from the debug
payload): whether the answer was served from cache, from which source, and how
stale it was. A warm repeat can be served from a cache the probe's own first
turn populated, and only that block says so.

Repeats hold the memory corpus fixed but not the transcript: every repeat is a
real turn that persists its own user and assistant message, so repeat N reads N
extra question/answer pairs as recent history. Each row reports
``transcript_messages_before_call`` so that growth is visible instead of
implied.

The prepared initial-context package is not held fixed either, and it changes
the PROMPT, not just the latency. A retained benchmark DB carries no current
package, and the engine refreshes them from inside the synchronous read, so the
first measured turn typically renders an empty package block and later repeats
render a populated one. Comparing those repeats as if they had measured the
same prompt is the mistake this probe exists to prevent, so every row carries
``state.initial_context_package``: whether a block was rendered, how many
tokens it cost, whether a refresh was requested, and the per-package read
status. Warming the package with an unmeasured turn was the alternative and was
rejected: it would persist an extra question/answer pair and destroy the one
thing repeat 0 is for, the engine's genuinely first turn.

The probe is additive: it never runs the judge (it measures latency, not
accuracy), it does not mutate the retained DB (it works on a temp snapshot),
and it does not touch any existing benchmark mode.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atagia import Atagia
from atagia.core.repositories import MessageRepository
from atagia.models.schemas_jobs import WorkerControlMode
from atagia.models.schemas_replay import AblationConfig
from atagia.services.chat_support import chat_model

from benchmarks.base import (
    BenchmarkConversation,
    BenchmarkDataset,
    BenchmarkQuestion,
)
from benchmarks.llm_config import provider_api_key_kwargs
from benchmarks.locomo.benchmark import (
    _BENCHMARK_CHARACTER_ID,
    _BENCHMARK_DB_FILENAME,
    _BENCHMARK_PLATFORM_ID,
    _BENCHMARK_USER_ID,
    _BENCHMARK_USER_PERSONA_ID,
    _DEFAULT_RETRIEVAL_PROFILE_ID,
    LoCoMoBenchmark,
)

_PROBE_NAME = "locomo_single_turn"


class SingleTurnProbeCacheState(BaseModel):
    """Observed context-cache state for one measured turn.

    Read from the engine's own debug payload rather than inferred: a warm repeat
    can be served from a cache this probe itself populated, which changes what
    the latency number means.

    Every field is required, so a key the engine stops emitting (or renames)
    fails validation instead of silently producing a partial label. Extra keys
    are tolerated: an engine that starts reporting MORE cache state is not a
    contract break, and failing on it would only make the probe brittle.
    """

    from_cache: bool
    cache_source: str | None
    staleness: float
    cache_age_seconds: float | None
    next_refresh_strategy: str
    need_detection_skipped: bool
    cache_key: str | None


class SingleTurnProbeInitialContextPackageState(BaseModel):
    """Prepared-context freshness for one measured turn.

    Two repeats that rendered different package blocks did not measure the same
    prompt. Recording the state per repeat is what makes the comparison
    interpretable rather than a latency number with a hidden variable in it.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    rendered: bool
    tokens_estimate: int = Field(ge=0)
    refresh_requested: bool
    # Package kind -> the read status the engine reported for it ("hit",
    # "miss", "stale", ...). Empty means the engine expected no package at all.
    package_statuses: dict[str, str] = Field(default_factory=dict)


class SingleTurnProbeState(BaseModel):
    """Explicit cache/warm labeling for one measured turn."""

    model_config = ConfigDict(extra="forbid")

    cold_runtime: bool
    repeat_index: int = Field(ge=0)
    turns_served_before_call: int = Field(ge=0)
    transcript_messages_before_call: int = Field(ge=0)
    cache: SingleTurnProbeCacheState
    initial_context_package: SingleTurnProbeInitialContextPackageState
    model_routing: dict[str, Any]


class SingleTurnProbeRow(BaseModel):
    """One raw per-repeat measurement (no aggregation)."""

    model_config = ConfigDict(extra="forbid")

    state: SingleTurnProbeState
    total_wall_ms: float = Field(ge=0.0)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieval_total_duration_ms: float | None = None
    response_text_chars: int = Field(ge=0)
    selected_memory_count: int = Field(ge=0)


class SingleTurnProbeReport(BaseModel):
    """Full output of a single-turn latency probe."""

    model_config = ConfigDict(extra="forbid")

    probe: str = _PROBE_NAME
    question_id: str
    conversation_id: str
    category: int
    concurrency: int = 1
    repeats: int = Field(ge=1)
    reuse_db_path: str
    model_routing: dict[str, Any]
    rows: list[SingleTurnProbeRow]


def _resolve_question(
    dataset: BenchmarkDataset,
    question_id: str,
) -> tuple[BenchmarkConversation, BenchmarkQuestion]:
    """Return the conversation and question for ``question_id`` or fail fast."""
    for conversation in dataset.conversations:
        for question in conversation.questions:
            if question.question_id == question_id:
                return conversation, question
    known = [
        question.question_id
        for conversation in dataset.conversations
        for question in conversation.questions
    ]
    raise ValueError(
        f"Unknown LoCoMo question id: {question_id!r}. "
        f"Known question ids: {', '.join(known) if known else '(none)'}"
    )


async def _conversation_message_count(engine: Atagia, *, conversation_id: str) -> int:
    """Count the persisted transcript a call is about to read as recent history."""
    runtime = engine.runtime
    if runtime is None:
        raise RuntimeError("Atagia runtime was unexpectedly unavailable")
    connection = await runtime.open_connection()
    try:
        messages = await MessageRepository(
            connection,
            runtime.clock,
        ).list_messages_for_conversation(conversation_id, _BENCHMARK_USER_ID)
    finally:
        await connection.close()
    return len(messages)


def _initial_context_package_state(
    debug: dict[str, Any],
) -> SingleTurnProbeInitialContextPackageState:
    """Read prepared-context freshness off the engine's own diagnostics.

    Fail fast on a missing block or a malformed package list: the probe would
    otherwise emit a row that reads as "no package" for a turn that rendered
    one, which is the exact confusion this field exists to remove.
    """
    diagnostics = debug.get("initial_context_package")
    if not isinstance(diagnostics, dict):
        raise ValueError(
            "single-turn probe requires the engine debug "
            "initial_context_package block to label prepared-context "
            "freshness; the engine debug contract changed"
        )
    packages = diagnostics.get("packages")
    if not isinstance(packages, list):
        raise ValueError(
            "single-turn probe requires initial_context_package.packages to be "
            "a list; the engine debug contract changed"
        )
    package_statuses: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError(
                "single-turn probe requires every "
                "initial_context_package.packages entry to be a mapping; the "
                "engine debug contract changed"
            )
        kind = str(package["package_kind"])
        if kind in package_statuses:
            raise ValueError(
                "single-turn probe cannot label a duplicated prepared-context "
                f"package kind: {kind}"
            )
        package_statuses[kind] = str(package["status"])
    return SingleTurnProbeInitialContextPackageState(
        enabled=bool(diagnostics["enabled"]),
        rendered=bool(diagnostics["rendered"]),
        tokens_estimate=int(diagnostics["tokens_estimate"]),
        refresh_requested=bool(diagnostics["refresh_requested"]),
        package_statuses=package_statuses,
    )


def _model_routing(benchmark: LoCoMoBenchmark, settings: Any) -> dict[str, Any]:
    """Describe the model routing actually in effect for the chat path."""
    routing: dict[str, Any] = {
        "provider": benchmark._llm_provider,
        "effective_chat_model": chat_model(settings),
    }
    routing.update(benchmark._atagia_model_kwargs())
    return routing


async def run_single_turn_probe(
    *,
    benchmark: LoCoMoBenchmark,
    question_id: str,
    reuse_db: str | Path | None,
    reuse_db_dir: str | Path | None,
    repeats: int = 1,
    ablation: AblationConfig | None = None,
    allow_untrusted_reuse: bool = False,
) -> SingleTurnProbeReport:
    """Measure one LoCoMo question against a retained DB with no parallelism.

    Fail fast: an unknown ``question_id`` or a missing/invalid retained DB
    raises immediately with a clear error and no fallback.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if reuse_db is None and reuse_db_dir is None:
        raise ValueError(
            "single-turn probe requires a retained DB via reuse_db or reuse_db_dir"
        )

    dataset = benchmark._adapter.load()
    conversation, question = _resolve_question(dataset, question_id)

    reuse_db_plan = benchmark._resolve_reuse_db_plan(
        reuse_db=reuse_db,
        reuse_db_dir=reuse_db_dir,
        selected_conversations=[conversation],
        allow_untrusted=allow_untrusted_reuse,
    )
    plan_entry = reuse_db_plan.get(conversation.conversation_id)
    if plan_entry is None:
        raise ValueError(
            "No retained benchmark DB found for conversation "
            f"{conversation.conversation_id!r} under the requested reuse path"
        )
    source_db_path = Path(plan_entry["db_path"])

    resolved_ablation = benchmark._benchmark_ablation(ablation)
    privacy_enforcement = benchmark._benchmark_privacy_enforcement(resolved_ablation)
    is_master = privacy_enforcement == "off"

    rows: list[SingleTurnProbeRow] = []
    with TemporaryDirectory(prefix="atagia-locomo-single-turn-") as temp_dir:
        snapshot_db_path = Path(temp_dir) / _BENCHMARK_DB_FILENAME
        # Snapshot the retained DB so the probe never mutates the source, and so
        # the memory corpus each repeat retrieves from is the ingested one,
        # matching the isolation the per-question scoring path uses. The
        # TRANSCRIPT is not held constant: every repeat is a real turn, so it
        # persists its own user and assistant message and repeat N reads N extra
        # question/answer pairs as recent history. Each row reports
        # `transcript_messages_before_call` so a comparison across repeats is
        # read against the transcript it actually ran on.
        benchmark._copy_sqlite_db(source_db_path, snapshot_db_path)
        # A single engine is built here: this construction IS the cold runtime.
        async with Atagia(
            db_path=snapshot_db_path,
            manifests_dir=benchmark._manifests_dir,
            **benchmark._atagia_model_kwargs(),
            **provider_api_key_kwargs(
                benchmark._llm_provider,
                benchmark._llm_api_key,
            ),
            **benchmark._inference_access_kwargs(),
            embedding_backend=benchmark._embedding_backend,
            embedding_model=benchmark._embedding_model,
            answer_postcondition_guard_enabled=(
                benchmark._answer_postcondition_guard_enabled
            ),
        ) as engine:
            runtime = engine.runtime
            if runtime is None:
                raise RuntimeError("Atagia runtime was unexpectedly unavailable")
            # Discard post-response memory work so repeats stay comparable and the
            # snapshot's memory corpus is not rewritten between turns.
            await engine.set_worker_control(
                WorkerControlMode.PAUSE_NEW_JOBS,
                reason="LoCoMo single-turn probe measures latency only",
            )
            model_routing = _model_routing(benchmark, runtime.settings)

            turns_served = 0
            for repeat_index in range(repeats):
                cold_runtime = turns_served == 0
                transcript_messages = await _conversation_message_count(
                    engine,
                    conversation_id=conversation.conversation_id,
                )
                wall_start = perf_counter()
                chat_result = await engine.chat(
                    user_id=_BENCHMARK_USER_ID,
                    conversation_id=conversation.conversation_id,
                    message=question.question_text,
                    mode=_DEFAULT_RETRIEVAL_PROFILE_ID,
                    ablation=resolved_ablation,
                    debug=True,
                    user_persona_id=_BENCHMARK_USER_PERSONA_ID,
                    platform_id=_BENCHMARK_PLATFORM_ID,
                    character_id=_BENCHMARK_CHARACTER_ID,
                    privacy_enforcement=privacy_enforcement,
                    authenticated_user_privilege_level=(
                        "atagia_master" if is_master else None
                    ),
                    authenticated_user_is_atagia_master=is_master,
                )
                total_wall_ms = (perf_counter() - wall_start) * 1000.0

                # Latency is the probe's whole purpose: a debug payload without
                # stage timings means the measurement contract broke — fail
                # fast instead of emitting an empty-timings row that reads as
                # a valid measurement.
                if not isinstance(chat_result.debug, dict):
                    raise ValueError(
                        "single-turn probe requires the engine debug payload; "
                        f"got {type(chat_result.debug).__name__}"
                    )
                debug = chat_result.debug
                stage_timings_raw = debug.get("stage_timings")
                if not isinstance(stage_timings_raw, dict) or not stage_timings_raw:
                    raise ValueError(
                        "single-turn probe requires non-empty debug stage_timings; "
                        "the engine debug contract changed"
                    )
                stage_timings_ms = {
                    str(key): float(value) for key, value in stage_timings_raw.items()
                }
                # CS-1.6 wants the cache state labeled, not guessed: the engine
                # already reports it, so a missing block means the debug
                # contract changed and the label would be a fabrication.
                cache_raw = debug.get("cache")
                if not isinstance(cache_raw, dict):
                    raise ValueError(
                        "single-turn probe requires the engine debug cache "
                        "block to label cache state; the engine debug contract "
                        "changed"
                    )
                cache_state = SingleTurnProbeCacheState.model_validate(cache_raw)
                # Prepared-context freshness, for the same reason the cache
                # block is read here: a repeat that rendered a package block and
                # one that did not are not two measurements of the same prompt.
                initial_context_package_state = _initial_context_package_state(debug)
                retrieval_trace = debug.get("retrieval_trace")
                retrieval_total_duration_ms = (
                    float(retrieval_trace["total_duration_ms"])
                    if isinstance(retrieval_trace, dict)
                    and isinstance(
                        retrieval_trace.get("total_duration_ms"), (int, float)
                    )
                    else None
                )
                selected_memory_count = (
                    len(chat_result.composed_context.selected_memory_ids)
                    if chat_result.composed_context is not None
                    else 0
                )

                rows.append(
                    SingleTurnProbeRow(
                        state=SingleTurnProbeState(
                            cold_runtime=cold_runtime,
                            repeat_index=repeat_index,
                            turns_served_before_call=turns_served,
                            transcript_messages_before_call=transcript_messages,
                            cache=cache_state,
                            initial_context_package=initial_context_package_state,
                            model_routing=model_routing,
                        ),
                        total_wall_ms=total_wall_ms,
                        stage_timings_ms=stage_timings_ms,
                        retrieval_total_duration_ms=retrieval_total_duration_ms,
                        response_text_chars=len(chat_result.response_text or ""),
                        selected_memory_count=selected_memory_count,
                    )
                )
                turns_served += 1

    return SingleTurnProbeReport(
        question_id=question.question_id,
        conversation_id=conversation.conversation_id,
        category=question.category,
        concurrency=1,
        repeats=repeats,
        reuse_db_path=str(source_db_path),
        model_routing=model_routing,
        rows=rows,
    )
