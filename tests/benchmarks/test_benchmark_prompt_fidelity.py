"""Benchmark Prompt Fidelity guard (CLAUDE.md -> "Benchmark Prompt Fidelity").

Every shadow/micro benchmark that exercises an engine component WITHOUT booting
the full Atagia engine must render the PRODUCTION ("champion") prompt imported
from ``src/atagia/...`` -- never a hand-copied duplicate. Duplicated prompts
silently drift from production (this is exactly how the extraction and
need_detection harness copies diverged), so the benchmark ends up measuring a
prompt the engine no longer uses.

This module is the automated enforcement: one test per card-family micro
benchmark asserts the benchmark's champion rendering path is byte-identical to
the engine's rendering path for a representative synthetic input. A reintroduced
duplicate -- or any drift between a benchmark champion prompt and the engine --
fails the build.

Two narrow, intentional exceptions are documented inline and explicitly NOT
asserted as champion-identity (they are not gaps):

1. Challenger prompts under evaluation -- alternative prompts compared
   side-by-side against the imported champion. These live in the benchmark by
   design until promoted/deleted. Here:
     * consequence ``build_json_prompt`` (legacy JSON challenger),
     * topic ``_build_*_v1.._v5`` (experimental route/boundary challengers),
     * language ``_JSON_*`` legacy JSON template (challenger).
2. A step production performs deterministically (no engine LLM prompt to
   import) -- a benchmark-only probe prompt is allowed. Here:
     * topic ``_build_artifact_prompt`` (the engine derives artifact links
       deterministically; there is no engine LLM counterpart).

Each per-family assertion renders NON-EMPTY prompt text and compares it against
the engine -- the guard is meaningful, not vacuous. Where the benchmark imports
the engine symbol directly (extraction, language), an ``is`` identity assertion
is used IN ADDITION to a rendered byte-identity check. Where the benchmark has
no standalone champion builder because it drives the engine object directly
(applicability, consequence, topic), the structural invariant (the benchmark
imports the engine and defines no local champion prompt builder) is asserted
alongside a non-empty engine rendering.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = PROJECT_ROOT / "manifests"


def _bench_settings(**overrides: Any) -> Settings:
    """Deterministic settings independent of the developer's local environment.

    Card examples default ON (the small/local target class), matching the engine
    default, so the rendered champion/engine prompts include the few-shot block.
    """
    base = Settings(
        sqlite_path=":memory:",
        migrations_path="./migrations",
        manifests_path="./manifests",
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        llm_component_models={},
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )
    return replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# Content-based anti-reintroduction guard (closes the name-agnostic blind spot).
# ---------------------------------------------------------------------------
# The original structural check only flagged a reintroduced champion duplicate
# when the duplicate's SYMBOL NAME contained "prompt". The need_detection
# duplicate this whole effort removed lived in an inline ``task = (...)`` block
# inside a function NOT named with "prompt" (``_naked_card_request``), so the
# same class of duplicate could slip back in under any function name.
#
# The durable invariant for these families is: a faithful benchmark IMPORTS the
# engine prompt; the engine's champion instruction/example text NEVER appears as
# a string literal inside the benchmark module source. A hand-copy -- inline
# ``task=(...)`` or any helper, regardless of symbol name -- reintroduces that
# literal and is caught here.
#
# Anchors are extracted at RUNTIME from the engine-rendered champion prompt so
# they stay in sync with the engine automatically. Each anchor is a distinctive
# CONTIGUOUS multi-line excerpt (full instruction sentences / example lines, not
# short generic fragments), chosen to avoid coincidental matches and -- where a
# documented challenger shares the champion's instruction head verbatim -- taken
# from the part of the champion the challenger does NOT reproduce.


def _benchmark_source_text(*module_relative_paths: str) -> str:
    """Concatenate the raw ``.py`` source of one benchmark family's modules."""
    parts: list[str] = []
    for relative in module_relative_paths:
        path = PROJECT_ROOT / relative
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _instruction_head_anchor(rendered_prompt: str, line_count: int = 2) -> str:
    """First ``line_count`` raw lines of a rendered champion prompt, contiguous.

    Blank lines are preserved so the anchor is a verbatim contiguous excerpt of
    the champion text (a hand-copy reproduces it exactly).
    """
    raw_lines = rendered_prompt.splitlines()
    anchor = "\n".join(raw_lines[:line_count])
    assert anchor in rendered_prompt, "instruction anchor must be present in the champion"
    assert anchor.strip(), "instruction anchor must be non-empty"
    return anchor


def _examples_block_anchor(rendered_prompt: str, line_count: int = 2) -> str:
    """Contiguous excerpt from the champion's ``Examples:`` block.

    Used where a documented challenger duplicates the champion instruction head
    verbatim but omits examples; the examples block is the distinguishing
    champion content a faithful benchmark never copies.
    """
    raw_lines = rendered_prompt.splitlines()
    marker_index = raw_lines.index("Examples:")
    anchor = "\n".join(raw_lines[marker_index + 1 : marker_index + 1 + line_count])
    assert anchor in rendered_prompt, "examples anchor must be present in the champion"
    assert anchor.strip(), "examples anchor must be non-empty"
    return anchor


def _assert_champion_text_absent(
    *,
    anchors: dict[str, str],
    benchmark_source: str,
    family: str,
) -> None:
    """Each champion anchor must be ABSENT from the benchmark source.

    Presence means the champion instruction/example text was hand-copied into
    the benchmark (an inline ``task=(...)`` duplicate or any local builder) --
    exactly the duplication class this guard prevents -- regardless of the
    enclosing symbol's name.
    """
    reintroduced = sorted(label for label, anchor in anchors.items() if anchor in benchmark_source)
    assert not reintroduced, (
        f"{family} benchmark reintroduced a champion card prompt as a source "
        f"literal (a faithful benchmark imports the engine prompt instead): "
        f"{reintroduced}"
    )


# ---------------------------------------------------------------------------
# 1. Memory extraction cards
# ---------------------------------------------------------------------------
# Champion path: the benchmark imports build_candidate_prompt / build_enrichment_prompt
#   straight from the engine and calls them (compare.py lines ~487, ~511).
# Engine path: atagia.memory.extraction_cards.build_candidate_prompt /
#   build_enrichment_prompt.
# Assertion: symbol identity (strongest) + rendered byte-identity, examples ON/OFF.
# No champion duplicate exists; the benchmark has only concurrency variants, no
# challenger prompt to exclude.
def _extraction_engine_prompts(include_examples: bool) -> str:
    from atagia.memory.extraction_cards import (
        CandidateDraft,
        build_candidate_prompt,
        build_enrichment_prompt,
    )
    from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
    from atagia.models.schemas_memory import ExtractionConversationContext

    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]
    resolved_policy = PolicyResolver().resolve(manifest, None, None)
    context = ExtractionConversationContext(
        user_id="usr_fidelity",
        conversation_id="cnv_fidelity",
        source_message_id="msg_fidelity",
        workspace_id="ws_fidelity",
        assistant_mode_id="coding_debug",
        recent_messages=[],
        privacy_enforcement="off",
    )
    candidate = CandidateDraft(candidate_id="cand_001", canonical_text="placeholder")
    prompts = [
        build_candidate_prompt(
            message_text="placeholder message",
            role="user",
            context=context,
            resolved_policy=resolved_policy,
            allowed_write_scopes=("chat", "character", "user"),
            occurred_at=None,
            prior_chunk_context=None,
            include_examples=include_examples,
        )
    ]
    for card in ("kind_scope", "evidence", "index", "temporal", "belief", "coverage_members"):
        prompts.append(
            build_enrichment_prompt(
                card,
                message_text="placeholder message",
                role="user",
                context=context,
                resolved_policy=resolved_policy,
                allowed_write_scopes=("chat", "character", "user"),
                occurred_at=None,
                prior_chunk_context=None,
                candidates=(candidate,),
                include_examples=include_examples,
            )
        )
    return "\n".join(prompts)


def test_memory_extraction_cards_use_engine_prompt() -> None:
    import benchmarks.memory_extraction_cards.compare as bench
    import atagia.memory.extraction_cards as engine

    # Identity: the benchmark's champion builders ARE the engine builders.
    assert bench.build_candidate_prompt is engine.build_candidate_prompt
    assert bench.build_enrichment_prompt is engine.build_enrichment_prompt
    assert bench._CARD_SYSTEM_PROMPTS is engine._CARD_SYSTEM_PROMPTS

    from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
    from atagia.models.schemas_memory import ExtractionConversationContext

    # Rendered byte-identity through both module references, examples ON and OFF.
    for include_examples in (True, False):
        engine_text = _extraction_engine_prompts(include_examples)
        assert engine_text.strip(), "engine extraction prompt must be non-empty"
        # Render the same cards through the benchmark's imported symbols.
        loader = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]
        policy = PolicyResolver().resolve(loader, None, None)
        context = ExtractionConversationContext(
            user_id="usr_fidelity",
            conversation_id="cnv_fidelity",
            source_message_id="msg_fidelity",
            workspace_id="ws_fidelity",
            assistant_mode_id="coding_debug",
            recent_messages=[],
            privacy_enforcement="off",
        )
        candidate = bench.CandidateDraft(
            candidate_id="cand_001", canonical_text="placeholder"
        )
        bench_prompts = [
            bench.build_candidate_prompt(
                message_text="placeholder message",
                role="user",
                context=context,
                resolved_policy=policy,
                allowed_write_scopes=("chat", "character", "user"),
                occurred_at=None,
                prior_chunk_context=None,
                include_examples=include_examples,
            )
        ]
        for card in ("kind_scope", "evidence", "index", "temporal", "belief", "coverage_members"):
            bench_prompts.append(
                bench.build_enrichment_prompt(
                    card,
                    message_text="placeholder message",
                    role="user",
                    context=context,
                    resolved_policy=policy,
                    allowed_write_scopes=("chat", "character", "user"),
                    occurred_at=None,
                    prior_chunk_context=None,
                    candidates=(candidate,),
                    include_examples=include_examples,
                )
            )
        assert "\n".join(bench_prompts) == engine_text


# ---------------------------------------------------------------------------
# 2. Need detection cards
# ---------------------------------------------------------------------------
# Champion path: the benchmark's _naked_card_request builds the request through
#   the engine NeedDetector._card_request (no hand-copy) for 7 "naked" cards.
# Engine path: NeedDetector._card_request invoked directly for the same cards.
# Assertion: rendered byte-identity of messages + max_output_tokens for all 7
#   naked cards. (The engine also has a "needs" card not graded by this shadow
#   benchmark; the 7 naked cards are exactly the benchmark's champion surface.)
# No champion duplicate: there is no parallel hand-copied prompt to exclude.
#
# Documented probe exception: the benchmark ALSO has a separate grouped 4-card
#   path (_card_request, phases phase1/2/3) with inline task strings for
#   language/memory_dependence/exactness/anchors. This grouping has NO engine
#   counterpart -- production splits exact/shape/facets/callback across distinct
#   single-purpose cards -- so there is no champion prompt to import or assert it
#   against. It is an experimental probe, not a hand-copy of a production prompt,
#   and is deliberately NOT gated here (parallel to the topic _build_artifact_prompt
#   probe note above). The engine-faithful surface is the naked path asserted below.
def test_need_detection_cards_use_engine_prompt() -> None:
    from benchmarks.need_detection_cards.__main__ import (
        _NAKED_CARD_NAMES,
        _authority_context_from_extraction_context,
        _build_context,
        _clock_for_case,
        _naked_card_request,
        _policy_for_case,
        CardModelSpec,
        NeedCardCase,
    )
    from atagia.memory.need_detector import NeedDetector

    settings = _bench_settings()
    case = NeedCardCase(
        case_id="fidelity_dummy",
        category="fidelity",
        query="PLACEHOLDER_QUERY_TOKEN",
        reference_time_iso="0001-01-01T00:00:00+00:00",
    )
    model = CardModelSpec(
        label="fidelity-dummy",
        model_spec="openrouter/provider/fidelity-dummy",
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    )

    detector = NeedDetector(
        llm_client=cast(Any, None),
        clock=_clock_for_case(case),
        settings=settings,
    )
    context = _build_context(case)
    authority = _authority_context_from_extraction_context(
        context, purpose="need_detection"
    )

    rendered_any = False
    for card_name in _NAKED_CARD_NAMES:
        champion = _naked_card_request(
            case=case, card_name=card_name, model=model, settings=settings
        )
        engine_request = detector._card_request(
            card_name=card_name,
            model=model.model_spec,
            message_text=case.query,
            role=case.role,
            context=context,
            resolved_policy=_policy_for_case(case),
            content_language_profile=list(case.content_language_profile),
            user_communication_profile=None,
            prompt_authority_context=authority,
        )
        champion_messages = [message.content for message in champion.messages]
        engine_messages = [message.content for message in engine_request.messages]
        assert all(text.strip() for text in champion_messages)
        assert champion_messages == engine_messages
        assert champion.max_output_tokens == engine_request.max_output_tokens
        rendered_any = True
    assert rendered_any, "need-detection naked card set must not be empty"


# ---------------------------------------------------------------------------
# 3. Consequence detection cards
# ---------------------------------------------------------------------------
# Champion path: the benchmark drives the engine ConsequenceDetector.detect()
#   directly (compare.py run_cards_variant), so the card prompts are the engine's
#   _card_task/_card_request -- there is no standalone champion builder in the
#   benchmark.
# Engine path: ConsequenceDetector._card_task for every card purpose.
# Assertion: structural invariant (benchmark imports the engine detector) +
#   non-empty engine rendering + CONTENT anti-reintroduction (each champion card
#   task's distinctive instruction text is ABSENT from the benchmark source, so a
#   name-agnostic inline ``task=(...)`` champion copy is caught).
# EXCLUDED exception (challenger, NOT a gap): compare.build_json_prompt /
#   _JSON_PROMPT_TEMPLATE is the legacy JSON challenger run side-by-side; it uses
#   DIFFERENT text from the champion cards, so it does not trip the content scan.
def test_consequence_detection_cards_use_engine_prompt() -> None:
    import benchmarks.consequence_detection_cards.compare as bench
    from atagia.memory import consequence_detector as engine
    from atagia.memory.consequence_detector import _CARD_PURPOSES, _card_task

    # Structural invariant: benchmark uses the engine detector, not a copy.
    assert bench.ConsequenceDetector is engine.ConsequenceDetector

    # Meaningful engine rendering: every card's champion task block is non-empty.
    rendered_cards = {
        card_name: _card_task(card_name, include_examples=True)
        for card_name in _CARD_PURPOSES
    }
    engine_prompt = "\n".join(rendered_cards.values())
    assert engine_prompt.strip(), "engine consequence card prompts must be non-empty"

    # Content anti-reintroduction: the champion instruction text for every card
    # must NOT appear as a literal in the benchmark source (a faithful benchmark
    # drives the engine detector and never hand-copies these prompts).
    anchors = {
        card_name: _instruction_head_anchor(rendered)
        for card_name, rendered in rendered_cards.items()
    }
    _assert_champion_text_absent(
        anchors=anchors,
        benchmark_source=_benchmark_source_text(
            "benchmarks/consequence_detection_cards/compare.py"
        ),
        family="consequence",
    )


# ---------------------------------------------------------------------------
# 4. Applicability cards
# ---------------------------------------------------------------------------
# Champion path: the benchmark drives the engine ApplicabilityScorer.score_shortlist
#   directly (compare.py), so card prompts come from the engine
#   ApplicabilityScorer._build_card_prompt -- no standalone champion builder.
# Engine path: ApplicabilityScorer._build_card_prompt rendered via score_shortlist
#   against a capturing provider (mirrors test_applicability_scorer._capture_card_prompts).
# Assertion: structural invariant (benchmark imports the engine scorer) +
#   non-empty engine-rendered card prompt + CONTENT anti-reintroduction (the
#   champion relevance/date card instruction text is ABSENT from the benchmark
#   source, so a name-agnostic inline ``task=(...)`` champion copy is caught).
# No challenger prompt to exclude (only concurrency/date-toggle variants).
def test_applicability_cards_use_engine_prompt() -> None:
    import asyncio

    import benchmarks.applicability_cards.compare as bench
    from atagia.memory import applicability_scorer as engine
    from atagia.memory.applicability_scorer import (
        APPLICABILITY_DATE_CARD_INSTRUCTION,
        APPLICABILITY_RELEVANCE_CARD_INSTRUCTION,
        ApplicabilityScorer,
    )
    from atagia.services.llm_client import LLMClient

    # Reuse the engine test's synthetic-case helpers (same shapes the leak-guard
    # test feeds through score_shortlist) so the rendered prompt is a faithful
    # engine production prompt, not a hand-built candidate dict.
    from tests.memory.test_applicability_scorer import (
        CannedApplicabilityCardProvider,
        _candidate,
        _context,
        _resolved_policy,
    )

    # Structural invariant: benchmark uses the engine scorer, not a copy.
    assert bench.ApplicabilityScorer is engine.ApplicabilityScorer

    provider = CannedApplicabilityCardProvider(
        relevance_output="candidate_000 exact",
        date_output="candidate_000 none",
    )
    scorer = ApplicabilityScorer(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        clock=FrozenClock(datetime(2026, 3, 30, 21, 0, tzinfo=timezone.utc)),
        settings=_bench_settings(),
    )
    asyncio.run(
        scorer.score_shortlist(
            [
                _candidate(
                    "mem_relative",
                    canonical_text="Yesterday I had a dentist appointment.",
                    rrf_score=0.7,
                )
            ],
            message_text="What appointment did I say I had yesterday?",
            conversation_context=_context(),
            resolved_policy=_resolved_policy(),
            detected_needs=[],
        )
    )
    assert provider.requests, "score_shortlist must issue at least one card request"
    rendered = provider.requests[0].messages[1].content
    assert rendered.strip(), "engine applicability card prompt must be non-empty"

    # Content anti-reintroduction: the champion relevance/date card instruction
    # text must NOT appear as a literal in the benchmark source (a faithful
    # benchmark drives the engine scorer and never hand-copies these prompts).
    _assert_champion_text_absent(
        anchors={
            "relevance": _instruction_head_anchor(
                APPLICABILITY_RELEVANCE_CARD_INSTRUCTION, line_count=3
            ),
            "date": _instruction_head_anchor(
                APPLICABILITY_DATE_CARD_INSTRUCTION, line_count=3
            ),
        },
        benchmark_source=_benchmark_source_text(
            "benchmarks/applicability_cards/compare.py"
        ),
        family="applicability",
    )


# ---------------------------------------------------------------------------
# 5. Topic working set cards
# ---------------------------------------------------------------------------
# Champion path: the benchmark calls the engine TopicWorkingSetUpdater._build_*
#   methods directly (compare.py _run_existing_route_card / _run_new_topic_track_card /
#   _run_content_cards / _run_boundary_cards) for the 4 DEFAULT cards -- no copy.
# Engine path: the same 4 TopicWorkingSetUpdater._build_* methods.
# Assertion: structural invariant (benchmark imports the engine updater) +
#   non-empty engine rendering of all 4 default cards + CONTENT anti-reintroduction
#   (each champion card's distinctive text is ABSENT from the benchmark source, so
#   a name-agnostic inline ``task=(...)`` champion copy is caught -- including one
#   named like a challenger).
# EXCLUDED exceptions (NOT gaps), verified NOT to trip the content scan:
#   * _build_*_v1.._v5 -- experimental route/boundary CHALLENGERS run side-by-side.
#     The new_topic_track challengers (v3/v4) duplicate the champion INSTRUCTION
#     head verbatim but omit the champion's examples block, so the anchor is taken
#     from the champion examples block they do NOT reproduce.
#   * _build_artifact_prompt -- benchmark-only probe for a step production performs
#     DETERMINISTICALLY (engine derives artifact links without an LLM prompt), so
#     there is no engine counterpart and its text matches no champion anchor.
def test_topic_working_set_cards_use_engine_prompt() -> None:
    import benchmarks.topic_working_set_cards.compare as bench
    from atagia.memory import topic_working_set as engine
    from atagia.memory.topic_working_set import (
        _TopicContent,
        _TopicRoute,
        TopicUpdateActionType,
        TopicWorkingSetUpdater,
    )
    from atagia.services.llm_client import (
        LLMClient,
        LLMCompletionRequest,
        LLMCompletionResponse,
        LLMEmbeddingRequest,
        LLMEmbeddingResponse,
        LLMProvider,
    )

    # Structural invariant: benchmark uses the engine updater, not a copy.
    assert bench.TopicWorkingSetUpdater is engine.TopicWorkingSetUpdater

    class _SilentProvider(LLMProvider):
        name = "topic-fidelity"

        async def complete(
            self, request: LLMCompletionRequest
        ) -> LLMCompletionResponse:
            raise AssertionError("no completion expected in topic fidelity test")

        async def embed(
            self, request: LLMEmbeddingRequest
        ) -> LLMEmbeddingResponse:
            raise AssertionError("embeddings unused in topic fidelity test")

    updater = TopicWorkingSetUpdater(
        llm_client=LLMClient(provider_name="topic-fidelity", providers=[_SilentProvider()]),
        clock=FrozenClock(datetime(2026, 4, 26, 2, 45, tzinfo=timezone.utc)),
        topic_repository=cast(Any, None),
        message_repository=cast(Any, None),
        settings=_bench_settings(),
    )
    snapshot = {"active_topics": [], "parked_topics": []}
    messages = [{"id": "msg_1", "seq": 1, "role": "user", "text": "Plan a budget for the move."}]
    route = _TopicRoute(
        action=TopicUpdateActionType.CREATE,
        target_id="tmp1",
        source_message_ids=("msg_1",),
    )
    content = _TopicContent(
        title="Moving budget", summary="The user is planning a moving budget."
    )
    default_prompts = {
        "existing_route": updater._build_existing_route_prompt(
            conversation_id="cnv_1", snapshot=snapshot, messages=messages
        ),
        "new_topic_track": updater._build_new_topic_track_prompt(
            conversation_id="cnv_1", snapshot=snapshot, messages=messages
        ),
        "content": updater._build_content_prompt(
            conversation_id="cnv_1",
            snapshot=snapshot,
            messages=messages,
            route=route,
            existing_topic=None,
        ),
        "boundary": updater._build_target_boundary_prompt(
            conversation_id="cnv_1", messages=messages, route=route, content=content
        ),
    }
    for name, prompt in default_prompts.items():
        assert prompt.strip(), f"engine topic card '{name}' must render non-empty"

    # Content anti-reintroduction: the champion card text must NOT appear as a
    # literal in the benchmark source (a faithful benchmark drives the engine
    # updater and never hand-copies these prompts). The anchor is taken from each
    # champion's examples block -- the part the new_topic_track challengers (which
    # duplicate the instruction head verbatim) deliberately omit -- so a genuine
    # champion copy is caught while the documented challengers are not.
    _assert_champion_text_absent(
        anchors={
            name: _examples_block_anchor(prompt)
            for name, prompt in default_prompts.items()
        },
        benchmark_source=_benchmark_source_text(
            "benchmarks/topic_working_set_cards/compare.py"
        ),
        family="topic",
    )


# ---------------------------------------------------------------------------
# 6. Language profile cards
# ---------------------------------------------------------------------------
# Champion path: the benchmark imports _card_prompt (aliased production_card_prompt)
#   from the engine and composes it with the engine compose_card_prompt
#   (compare.py run_card_variant lines ~439-450).
# Engine path: atagia.memory.language_profile._card_prompt + card_prompt.compose_card_prompt.
# Assertion: symbol identity (strongest) + rendered byte-identity of the composed
#   prompt for every card, examples ON and OFF.
# EXCLUDED exception (challenger, NOT a gap): the benchmark's _JSON_* legacy JSON
#   template is the JSON challenger run side-by-side; not asserted to match engine.
def test_language_profile_cards_use_engine_prompt() -> None:
    import benchmarks.language_profile_cards.compare as bench
    from atagia.memory import language_profile as engine
    from atagia.memory.card_prompt import compose_card_prompt
    from atagia.memory.language_profile import _CARD_NAMES, _card_prompt

    # Identity: the benchmark's champion builder IS the engine builder.
    assert bench.production_card_prompt is engine._card_prompt
    assert bench.compose_card_prompt is compose_card_prompt

    for include_examples in (True, False):
        rendered_any = False
        for card_name in _CARD_NAMES:
            engine_instruction, engine_examples = _card_prompt(
                card_name=card_name,
                message_text="PLACEHOLDER_QUERY_TOKEN",
                role="user",
            )
            engine_text = compose_card_prompt(
                engine_instruction,
                engine_examples,
                include_examples=include_examples,
            )
            bench_instruction, bench_examples = bench.production_card_prompt(
                card_name=card_name,
                message_text="PLACEHOLDER_QUERY_TOKEN",
                role="user",
            )
            bench_text = bench.compose_card_prompt(
                bench_instruction,
                bench_examples,
                include_examples=include_examples,
            )
            assert engine_text.strip(), "engine language card prompt must be non-empty"
            assert bench_text == engine_text
            rendered_any = True
        assert rendered_any, "language card set must not be empty"


# ---------------------------------------------------------------------------
# 7. Compactor segmentation cards (card 1 = ranges, card 2 = per-range summary)
# ---------------------------------------------------------------------------
# Card 1 path: benchmarks/model_casting/roles.py imports the engine range-card
#   constants and renders them; it must use the SAME objects, not copies.
# Card 2 path: no benchmark builds the summary card -- the card2_summary harness
#   drives the engine method Compactor._summarize_message_ranges_card directly --
#   so the card 2 prompt text must never appear as a literal in any benchmark.
# Assertion: symbol identity for card 1 + CONTENT anti-reintroduction for both
#   cards across every benchmark source file.
def _all_benchmark_source_text() -> str:
    parts = [
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "benchmarks").rglob("*.py"))
    ]
    return "\n".join(parts)


def test_compactor_segmentation_cards_use_engine_prompt() -> None:
    import benchmarks.model_casting.roles as roles
    from atagia.memory import compactor as engine

    # Card 1 structural invariant: the model-casting mirror imports the engine
    # range-card constants and renderer, not local copies.
    assert roles._SEGMENTATION_RANGE_CARD_HEAD is engine._SEGMENTATION_RANGE_CARD_HEAD
    assert roles._SEGMENTATION_RANGE_CARD_TAIL is engine._SEGMENTATION_RANGE_CARD_TAIL
    assert roles._SEGMENTATION_RANGE_CARD_EXAMPLES is engine._SEGMENTATION_RANGE_CARD_EXAMPLES
    assert roles.Compactor is engine.Compactor

    benchmark_source = _all_benchmark_source_text()

    # Content anti-reintroduction: neither card's distinctive instruction head
    # nor its examples head may be hand-copied into any benchmark source.
    anchors = {
        "range_head": _instruction_head_anchor(engine._SEGMENTATION_RANGE_CARD_HEAD),
        "range_examples": _instruction_head_anchor(engine._SEGMENTATION_RANGE_CARD_EXAMPLES),
        "summary_head": _instruction_head_anchor(engine._RANGE_SUMMARY_CARD_HEAD),
        "summary_examples": _instruction_head_anchor(engine._RANGE_SUMMARY_CARD_EXAMPLES),
    }
    _assert_champion_text_absent(
        anchors=anchors,
        benchmark_source=benchmark_source,
        family="compactor_segmentation",
    )
