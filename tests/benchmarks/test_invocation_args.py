"""CLI credentials must never reach a persisted benchmark artifact.

Both harnesses record their invocation verbatim so a run is reproducible, and
both accept ``--api-key``. The Settings block redacts the very same credential,
so an unredacted argv copy sitting beside it defeats the redaction entirely.

These tests exercise the argv path itself: they build the REAL parsers, redact a
realistic invocation, and then assert the persisted report and manifest keep the
flag (so the invocation stays auditable) while the secret is gone.

They cover both ways a credential reached an artifact: the abbreviated spellings
argparse accepts but literal option matching missed, and the outer launchers
(``full_runner``, ``retained_slice_runner``), which build the credential-bearing
command themselves and persisted it without going through the redaction the
inner CLIs apply to their own argv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from atagia.core.effective_settings import REDACTED_EMPTY, REDACTED_SET
from benchmarks.atagia_bench import __main__ as atagia_bench_cli
from benchmarks.atagia_bench.runner import AtagiaBenchRunner
from benchmarks.invocation_args import (
    credential_option_strings,
    redact_invocation_args,
)
from benchmarks.locomo import __main__ as locomo_cli
from benchmarks.locomo.benchmark import LoCoMoBenchmark
from benchmarks.locomo.full_runner import (
    FullLoCoMoRunConfig,
    build_ingest_command,
    run_full_locomo,
)
from benchmarks.locomo.retained_slice_runner import (
    RetainedLoCoMoJob,
    RetainedLoCoMoRunConfig,
    build_locomo_command,
    redacted_command,
    run_jobs,
)
from tests.benchmarks.test_atagia_bench_manifest import (
    _write_minimal_atagia_bench_data,
)
from tests.benchmarks.test_locomo_benchmark import (
    MANIFESTS_DIR,
    BenchmarkProvider,
    _install_stub_client,
    _write_dataset,
)

_SECRET = "sk-live-DEADBEEF-THIS-IS-A-SECRET"

_PARSER_BUILDERS = {
    "locomo": locomo_cli._build_parser,
    "atagia_bench": atagia_bench_cli._build_parser,
}

_ATAGIA_BENCH_QUESTION = {
    "question_id": "mini-q1",
    "question_text": "What color is the notebook?",
    "ground_truth": "red",
    "answer_type": "exact_match",
    "category_tags": ["factual"],
    "evidence_turn_ids": ["mini-t1"],
    "grader": "exact_match",
}


@pytest.mark.parametrize("harness", sorted(_PARSER_BUILDERS))
def test_credential_flags_are_derived_from_the_parser(harness: str) -> None:
    """The credential set comes from the parser, not a hand-kept list: any
    option whose dest is secret-shaped is covered the day it is added."""
    parser = _PARSER_BUILDERS[harness]()

    credentials = credential_option_strings(parser)

    assert "--api-key" in credentials
    # Token COUNT flags name-match "token" but are not credentials.
    assert not {option for option in credentials if "tokens" in option}


@pytest.mark.parametrize("harness", sorted(_PARSER_BUILDERS))
def test_both_argparse_spellings_are_redacted(harness: str) -> None:
    """``--api-key VALUE`` and ``--api-key=VALUE`` both leave the process."""
    parser = _PARSER_BUILDERS[harness]()

    spaced = redact_invocation_args(
        ["--api-key", _SECRET, "--provider", "openai"], parser
    )
    joined = redact_invocation_args(
        [f"--api-key={_SECRET}", "--provider", "openai"], parser
    )
    empty = redact_invocation_args(["--api-key="], parser)

    assert spaced == ["--api-key", REDACTED_SET, "--provider", "openai"]
    assert joined == [f"--api-key={REDACTED_SET}", "--provider", "openai"]
    assert empty == [f"--api-key={REDACTED_EMPTY}"]
    # A non-credential flag whose value merely looks like one is untouched.
    assert redact_invocation_args(["--provider", _SECRET], parser) == [
        "--provider",
        _SECRET,
    ]


def test_redaction_survives_a_value_that_looks_like_a_flag() -> None:
    """The value after a credential flag is redacted whatever it looks like."""
    parser = locomo_cli._build_parser()

    assert redact_invocation_args(["--api-key", "--provider"], parser) == [
        "--api-key",
        REDACTED_SET,
    ]


@pytest.mark.parametrize("harness", sorted(_PARSER_BUILDERS))
@pytest.mark.parametrize("abbreviation", ["--api-k", "--api"])
def test_abbreviated_credential_flags_are_redacted(
    harness: str,
    abbreviation: str,
) -> None:
    """argparse accepts any unambiguous prefix of a long option, so a run really
    can be launched with ``--api-k <secret>``.

    Matching option strings literally let every abbreviation through verbatim
    while the parser happily consumed the key, so the artifact carried the
    credential in cleartext. The parser is asked here whether it accepts the
    spelling, so the test fails both if argparse stops accepting it and if the
    redaction stops covering it.
    """
    parser = _PARSER_BUILDERS[harness]()
    parsed = parser.parse_args([abbreviation, _SECRET, "--provider", "openai"])
    assert parsed.api_key == _SECRET

    spaced = redact_invocation_args([abbreviation, _SECRET], parser)
    joined = redact_invocation_args([f"{abbreviation}={_SECRET}"], parser)

    assert spaced == [abbreviation, REDACTED_SET]
    assert joined == [f"{abbreviation}={REDACTED_SET}"]


@pytest.mark.parametrize("harness", sorted(_PARSER_BUILDERS))
def test_an_ambiguous_credential_prefix_still_fails_closed(harness: str) -> None:
    """``--a`` matches several options, so argparse rejects the invocation.

    The value never reaches the parser, but it does reach anything that records
    the argv on its way there, so resolution fails closed rather than reasoning
    about which spellings a run can survive.
    """
    parser = _PARSER_BUILDERS[harness]()
    with pytest.raises(SystemExit):
        parser.parse_args(["--a", _SECRET])

    assert redact_invocation_args(["--a", _SECRET], parser) == ["--a", REDACTED_SET]


def test_an_exact_non_credential_option_keeps_its_value() -> None:
    """Prefix expansion must not swallow real options: argparse resolves an
    exact option string to that option, so a flag that merely happens to prefix
    a credential one still records its value."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--api")
    parser.add_argument("--api-key")

    assert redact_invocation_args(["--api", "https://gateway.example"], parser) == [
        "--api",
        "https://gateway.example",
    ]
    assert redact_invocation_args(["--api-k", _SECRET], parser) == [
        "--api-k",
        REDACTED_SET,
    ]


def test_end_of_options_marker_is_not_treated_as_an_abbreviation() -> None:
    """``--`` is argparse's end-of-options marker, not a prefix of every long
    option, so it must not redact the argument that follows it."""
    parser = locomo_cli._build_parser()

    assert redact_invocation_args(["--", "positional"], parser) == ["--", "positional"]


@pytest.mark.asyncio
async def test_locomo_manifest_keeps_the_flag_and_drops_the_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end for LoCoMo: the persisted invocation still shows that an API
    key was supplied, and the key itself is not in the artifact."""
    _install_stub_client(monkeypatch, BenchmarkProvider())
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key=_SECRET,
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )
    argv = ["--api-key", _SECRET, "--provider", "openai", "--parallel-questions", "1"]

    report = await benchmark.run(
        invocation_args=redact_invocation_args(argv, locomo_cli._build_parser()),
    )

    persisted = report.model_info["invocation_args"]
    assert persisted == [
        "--api-key",
        REDACTED_SET,
        "--provider",
        "openai",
        "--parallel-questions",
        "1",
    ]

    report_path = tmp_path / "locomo-report.json"
    report_path.write_bytes(b'{"benchmark_name": "LoCoMo"}')
    manifest = benchmark.build_run_manifest(report, report_path=report_path)

    serialized = json.dumps(manifest)
    assert "--api-key" in serialized
    assert _SECRET not in serialized


@pytest.mark.asyncio
async def test_atagia_bench_report_keeps_the_flag_and_drops_the_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end for Atagia-bench, whose report persists ``invocation_args``
    under ``config`` and carries it into the run manifest."""
    _install_stub_client(monkeypatch, BenchmarkProvider())
    runner = AtagiaBenchRunner(
        llm_provider="openai",
        llm_api_key=_SECRET,
        llm_model="answer-model",
        judge_model="judge-model",
        data_dir=_write_minimal_atagia_bench_data(
            tmp_path,
            question=_ATAGIA_BENCH_QUESTION,
        ),
    )
    argv = [f"--api-key={_SECRET}", "--provider", "openai"]

    report = await runner.run(
        benchmark_db_dir=tmp_path / "dbs",
        allow_temp_benchmark_db_dir=True,
        invocation_args=redact_invocation_args(
            argv,
            atagia_bench_cli._build_parser(),
        ),
    )

    assert report.config["invocation_args"] == [
        f"--api-key={REDACTED_SET}",
        "--provider",
        "openai",
    ]

    report_path = tmp_path / "atagia-bench-report.json"
    report_path.write_text(json.dumps({"benchmark_name": "Atagia-bench"}))
    manifest = runner.build_run_manifest(report, report_path=report_path)

    serialized = json.dumps(manifest)
    assert "--api-key" in serialized
    assert _SECRET not in serialized


def test_full_runner_manifest_never_persists_the_api_key(tmp_path: Path) -> None:
    """The outer launcher writes the command it ran into its manifest.

    It builds ``--api-key <secret>`` itself and persisted it four times (two
    phases, each as a list and as shell text), which defeated the inner CLI's
    redaction completely: the same run produced a report with the key redacted
    and a manifest with it in cleartext.
    """
    config = FullLoCoMoRunConfig(
        data_path=tmp_path / "locomo.json",
        output_dir=tmp_path / "out",
        db_dir=tmp_path / "dbs",
        provider="openai",
        api_key=_SECRET,
        dry_run=True,
    )

    manifest_path = run_full_locomo(config)

    serialized = manifest_path.read_text(encoding="utf-8")
    assert _SECRET not in serialized
    manifest = json.loads(serialized)
    assert [phase["phase"] for phase in manifest["phases"]] == ["ingest", "evaluate"]
    for phase in manifest["phases"]:
        assert "--api-key" in phase["command"]
        assert phase["command"][phase["command"].index("--api-key") + 1] == REDACTED_SET
        assert REDACTED_SET in phase["command_text"]


def test_full_runner_still_passes_the_real_key_to_the_child(tmp_path: Path) -> None:
    """Redaction is for the artifact only: the phase command that actually runs
    must still carry the credential, or the redaction would break the run."""
    config = FullLoCoMoRunConfig(
        data_path=tmp_path / "locomo.json",
        output_dir=tmp_path / "out",
        db_dir=tmp_path / "dbs",
        provider="openai",
        api_key=_SECRET,
        dry_run=True,
    )

    command = build_ingest_command(config)

    assert command[command.index("--api-key") + 1] == _SECRET


@pytest.mark.asyncio
async def test_retained_slice_runner_manifest_never_persists_the_api_key(
    tmp_path: Path,
) -> None:
    """The shard runner persists ``base_args`` verbatim plus every command built
    from them, and those args are hand-written config -- abbreviated spellings
    included, since nothing normalizes them before they reach the child."""
    base_args = [
        "--data-path",
        str(tmp_path / "locomo.json"),
        "--api-k",
        _SECRET,
    ]
    config = RetainedLoCoMoRunConfig(
        base_args=base_args,
        jobs=[
            RetainedLoCoMoJob(
                name="shard-a",
                output=tmp_path / "shard-a",
                conversations=["conv-a"],
                questions=["conv-a:q1"],
            )
        ],
    )

    shards = await run_jobs(config, max_workers=1, dry_run=True)

    assert _SECRET not in json.dumps(shards)
    command = shards[0]["command"]
    assert command[command.index("--api-k") + 1] == REDACTED_SET
    # The command the shard actually runs is untouched.
    assert _SECRET in build_locomo_command(config.jobs[0], base_args=base_args)
    assert _SECRET not in json.dumps(redacted_command(base_args))
