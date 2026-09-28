"""Tests for the LoCoMo single-turn latency probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from atagia import Atagia
from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from benchmarks.locomo.__main__ import _build_parser
from benchmarks.locomo.benchmark import LoCoMoBenchmark
from benchmarks.locomo.single_turn_probe import (
    SingleTurnProbeReport,
    run_single_turn_probe,
)
from tests.benchmarks.test_locomo_benchmark import (
    MANIFESTS_DIR,
    BenchmarkProvider,
    _install_stub_client,
    _write_dataset,
)

_ASTROLABE_QUESTION_ID = "conv-test-1:q1"


class ProbeBenchmarkProvider(BenchmarkProvider):
    """Benchmark stub plus the context-cache staleness signal warm turns hit.

    A warm repeat reuses the same engine, so its second turn scores the context
    cache from the first turn. That path calls the staleness signal detector,
    a purpose the per-question cold benchmark path never exercises.
    """

    async def complete(
        self, request: LLMCompletionRequest
    ) -> LLMCompletionResponse:
        if request.metadata.get("purpose") == "context_cache_signal_detection":
            return self._response(
                request,
                json.dumps(
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
        return await super().complete(request)


async def _ingest_retained_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, LoCoMoBenchmark]:
    """Ingest the mini dataset and return the retained DB path plus a benchmark."""
    provider = ProbeBenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    db_dir = tmp_path / "dbs"
    ingest_benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )
    await ingest_benchmark.run(
        conversation_ids=["conv-test-1"],
        categories=[1],
        max_questions=1,
        benchmark_db_dir=db_dir,
        allow_temp_benchmark_db_dir=True,
        keep_db=True,
        ingest_mode="bulk",
    )
    retained_dbs = list(db_dir.glob("*/benchmark.db"))
    assert len(retained_dbs) == 1
    probe_benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )
    return retained_dbs[0], probe_benchmark


@pytest.mark.asyncio
async def test_single_turn_probe_labels_cold_then_warm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_db, probe_benchmark = await _ingest_retained_db(tmp_path, monkeypatch)

    report = await run_single_turn_probe(
        benchmark=probe_benchmark,
        question_id=_ASTROLABE_QUESTION_ID,
        reuse_db=retained_db,
        reuse_db_dir=None,
        repeats=2,
    )

    assert isinstance(report, SingleTurnProbeReport)
    assert report.probe == "locomo_single_turn"
    assert report.question_id == _ASTROLABE_QUESTION_ID
    assert report.conversation_id == "conv-test-1"
    assert report.category == 1
    # No parallelism: the probe hard-sets concurrency to 1.
    assert report.concurrency == 1
    assert report.repeats == 2
    assert Path(report.reuse_db_path) == retained_db
    assert report.model_routing["provider"] == "openai"
    assert report.model_routing["effective_chat_model"]

    assert len(report.rows) == 2
    cold_row, warm_row = report.rows

    # Cold/warm is derived from the engine's served-turn count, not the index.
    assert cold_row.state.repeat_index == 0
    assert cold_row.state.cold_runtime is True
    assert cold_row.state.turns_served_before_call == 0
    assert cold_row.state.model_routing["provider"] == "openai"

    assert warm_row.state.repeat_index == 1
    assert warm_row.state.cold_runtime is False
    assert warm_row.state.turns_served_before_call == 1

    # Engine age is not cache state: the cache block is read from the engine's
    # own debug payload, so a warm repeat served from the cache the first turn
    # populated says so instead of hiding behind `cold_runtime=False`.
    assert cold_row.state.cache.from_cache is False
    assert isinstance(warm_row.state.cache.from_cache, bool)
    assert warm_row.state.cache.cache_key

    # Repeats are real turns: each one persists a user and an assistant message,
    # so the transcript a repeat reads as recent history is NOT the one the
    # previous repeat read. The rows report that instead of implying otherwise.
    assert warm_row.state.transcript_messages_before_call == (
        cold_row.state.transcript_messages_before_call + 2
    )

    for row in report.rows:
        assert row.total_wall_ms > 0.0
        # The retrieval trace surfaces per-stage timings; at least one stage ran.
        assert row.stage_timings_ms
        assert all(value >= 0.0 for value in row.stage_timings_ms.values())
        assert any(value > 0.0 for value in row.stage_timings_ms.values())


@pytest.mark.asyncio
async def test_single_turn_probe_single_repeat_is_cold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_db, probe_benchmark = await _ingest_retained_db(tmp_path, monkeypatch)

    report = await run_single_turn_probe(
        benchmark=probe_benchmark,
        question_id=_ASTROLABE_QUESTION_ID,
        reuse_db=retained_db,
        reuse_db_dir=None,
        repeats=1,
    )

    assert len(report.rows) == 1
    assert report.rows[0].state.cold_runtime is True
    assert report.rows[0].state.turns_served_before_call == 0
    # The JSON payload carries the full state block for the founder to read.
    payload = report.model_dump(mode="json")
    assert payload["concurrency"] == 1
    state_payload = payload["rows"][0]["state"]
    assert state_payload["cold_runtime"] is True
    assert "model_routing" in state_payload
    assert state_payload["cache"]["from_cache"] is False
    assert "cache_source" in state_payload["cache"]
    assert state_payload["transcript_messages_before_call"] >= 0


@pytest.mark.asyncio
async def test_single_turn_probe_labels_prepared_context_freshness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeats that render different package blocks must say so.

    The engine refreshes prepared-context packages from inside the synchronous
    read, so a retained DB makes repeat 0 render one thing and repeat 1 another.
    Latency rows that hide that are comparing two different prompts.
    """
    retained_db, probe_benchmark = await _ingest_retained_db(tmp_path, monkeypatch)
    original_chat = Atagia.chat
    engine_diagnostics: list[dict[str, object]] = []

    async def capture_package_diagnostics(
        self: Atagia,
        *args: object,
        **kwargs: object,
    ):
        result = await original_chat(self, *args, **kwargs)
        assert isinstance(result.debug, dict)
        engine_diagnostics.append(dict(result.debug["initial_context_package"]))
        return result

    monkeypatch.setattr(Atagia, "chat", capture_package_diagnostics)

    report = await run_single_turn_probe(
        benchmark=probe_benchmark,
        question_id=_ASTROLABE_QUESTION_ID,
        reuse_db=retained_db,
        reuse_db_dir=None,
        repeats=2,
    )

    assert len(engine_diagnostics) == 2
    for row, diagnostics in zip(report.rows, engine_diagnostics, strict=True):
        package_state = row.state.initial_context_package
        # Reported, not inferred: each repeat carries the prepared-context
        # state the engine itself produced for that turn.
        assert package_state.enabled is diagnostics["enabled"]
        assert package_state.rendered is diagnostics["rendered"]
        assert package_state.tokens_estimate == diagnostics["tokens_estimate"]
        assert package_state.refresh_requested is diagnostics["refresh_requested"]
        assert package_state.package_statuses == {
            str(package["package_kind"]): str(package["status"])
            for package in diagnostics["packages"]
        }
    # The founder reads the JSON artifact, so the block has to survive the dump.
    payload = report.model_dump(mode="json")
    for index, row in enumerate(report.rows):
        assert payload["rows"][index]["state"]["initial_context_package"] == (
            row.state.initial_context_package.model_dump(mode="json")
        )


@pytest.mark.asyncio
async def test_single_turn_probe_fails_fast_without_prepared_context_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prepared-context label the engine did not report would be a guess."""
    retained_db, probe_benchmark = await _ingest_retained_db(tmp_path, monkeypatch)
    original_chat = Atagia.chat

    async def chat_without_package_block(self: Atagia, *args: object, **kwargs: object):
        result = await original_chat(self, *args, **kwargs)
        assert isinstance(result.debug, dict)
        result.debug.pop("initial_context_package")
        return result

    monkeypatch.setattr(Atagia, "chat", chat_without_package_block)

    with pytest.raises(ValueError, match="initial_context_package block"):
        await run_single_turn_probe(
            benchmark=probe_benchmark,
            question_id=_ASTROLABE_QUESTION_ID,
            reuse_db=retained_db,
            reuse_db_dir=None,
            repeats=1,
        )


@pytest.mark.asyncio
async def test_single_turn_probe_fails_fast_without_engine_cache_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache label the engine did not report would be a fabrication."""
    retained_db, probe_benchmark = await _ingest_retained_db(tmp_path, monkeypatch)
    original_chat = Atagia.chat

    async def chat_without_cache_block(self: Atagia, *args: object, **kwargs: object):
        result = await original_chat(self, *args, **kwargs)
        assert isinstance(result.debug, dict)
        result.debug.pop("cache")
        return result

    monkeypatch.setattr(Atagia, "chat", chat_without_cache_block)

    with pytest.raises(ValueError, match="debug cache block"):
        await run_single_turn_probe(
            benchmark=probe_benchmark,
            question_id=_ASTROLABE_QUESTION_ID,
            reuse_db=retained_db,
            reuse_db_dir=None,
            repeats=1,
        )


@pytest.mark.asyncio
async def test_single_turn_probe_unknown_question_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_db, probe_benchmark = await _ingest_retained_db(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="Unknown LoCoMo question id"):
        await run_single_turn_probe(
            benchmark=probe_benchmark,
            question_id="conv-test-1:q999",
            reuse_db=retained_db,
            reuse_db_dir=None,
            repeats=1,
        )


@pytest.mark.asyncio
async def test_single_turn_probe_requires_retained_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _retained_db, probe_benchmark = await _ingest_retained_db(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="requires a retained DB"):
        await run_single_turn_probe(
            benchmark=probe_benchmark,
            question_id=_ASTROLABE_QUESTION_ID,
            reuse_db=None,
            reuse_db_dir=None,
            repeats=1,
        )


@pytest.mark.asyncio
async def test_single_turn_probe_rejects_zero_repeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_db, probe_benchmark = await _ingest_retained_db(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="repeats must be at least 1"):
        await run_single_turn_probe(
            benchmark=probe_benchmark,
            question_id=_ASTROLABE_QUESTION_ID,
            reuse_db=retained_db,
            reuse_db_dir=None,
            repeats=0,
        )


def test_cli_parses_single_turn_probe_flags() -> None:
    args = _build_parser().parse_args(
        [
            "--data-path",
            "benchmarks/data/locomo10.json",
            "--provider",
            "openai",
            "--answer-model",
            "openai/chat-latest",
            "--reuse-db",
            "dbs/conv-test-1",
            "--single-turn-probe",
            "--question",
            "conv-test-1:q1",
            "--repeats",
            "3",
            "--single-turn-probe-output",
            "out/probe.json",
        ]
    )

    assert args.single_turn_probe is True
    assert args.question == "conv-test-1:q1"
    assert args.repeats == 3
    assert args.reuse_db == "dbs/conv-test-1"
    assert args.single_turn_probe_output == "out/probe.json"


def test_cli_probe_defaults() -> None:
    args = _build_parser().parse_args(
        [
            "--data-path",
            "benchmarks/data/locomo10.json",
            "--provider",
            "openai",
            "--answer-model",
            "openai/chat-latest",
        ]
    )

    assert args.single_turn_probe is False
    assert args.question is None
    assert args.repeats == 1
    assert args.single_turn_probe_output is None


@pytest.mark.asyncio
async def test_single_turn_probe_never_mutates_retained_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe works on a temp snapshot; the retained DB stays byte-identical."""
    retained_db, probe_benchmark = await _ingest_retained_db(tmp_path, monkeypatch)
    digest_before = hashlib.sha256(retained_db.read_bytes()).hexdigest()

    await run_single_turn_probe(
        benchmark=probe_benchmark,
        question_id=_ASTROLABE_QUESTION_ID,
        reuse_db=retained_db,
        reuse_db_dir=None,
        repeats=2,
    )

    digest_after = hashlib.sha256(retained_db.read_bytes()).hexdigest()
    assert digest_after == digest_before
    sidecars = [
        sidecar
        for suffix in ("-wal", "-shm")
        if (sidecar := retained_db.with_name(retained_db.name + suffix)).exists()
    ]
    assert not sidecars
