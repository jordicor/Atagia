from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.locomo.retained_slice_runner import (
    RetainedLoCoMoJob,
    build_locomo_command,
    latest_report_path,
    load_config,
)


def test_load_config_accepts_retained_conversation_jobs(tmp_path: Path) -> None:
    config_path = tmp_path / "slice.json"
    config_path.write_text(
        json.dumps(
            {
                "base_args": ["--data-path", "benchmarks/data/locomo10.json"],
                "aggregate_output": "benchmarks/results/slice/combined/report.json",
                "jobs": [
                    {
                        "name": "fixture-retained",
                        "output": "benchmarks/results/slice/fixture-retained",
                        "conversation": "fixture-retained",
                        "questions": ["fixture-retained:q2", "fixture-retained:q4"],
                        "reuse_db": "docs/tmp/dbs/fixture-retained",
                        "extra_args": ["--trusted-evaluation"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.base_args == ["--data-path", "benchmarks/data/locomo10.json"]
    assert config.aggregate_output == Path("benchmarks/results/slice/combined/report.json")
    assert len(config.jobs) == 1
    job = config.jobs[0]
    assert job.name == "fixture-retained"
    assert job.conversations == ["fixture-retained"]
    assert job.questions == ["fixture-retained:q2", "fixture-retained:q4"]
    assert job.reuse_db == Path("docs/tmp/dbs/fixture-retained")
    assert job.extra_args == ["--trusted-evaluation"]


def test_build_locomo_command_uses_reuse_db_evaluate_only() -> None:
    job = RetainedLoCoMoJob(
        name="typed-fixture",
        output=Path("out/typed/fixture-typed"),
        conversations=["fixture-typed"],
        questions=["fixture-typed:q4"],
        reuse_db=Path("dbs/fixture-typed"),
        extra_args=["--ablation", '{"enable_typed_relation_recall": true}'],
    )

    command = build_locomo_command(
        job,
        base_args=["--provider", "openrouter"],
        python_executable="python",
    )

    assert command == [
        "python",
        "-m",
        "benchmarks.locomo",
        "--provider",
        "openrouter",
        "--output",
        "out/typed/fixture-typed",
        "--conversations",
        "fixture-typed",
        "--questions",
        "fixture-typed:q4",
        "--reuse-db",
        "dbs/fixture-typed",
        "--evaluate-only",
        "--ablation",
        '{"enable_typed_relation_recall": true}',
    ]


def test_latest_report_path_requires_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        latest_report_path(tmp_path)

    older = tmp_path / "locomo-report-20260510T010000Z.json"
    newer = tmp_path / "locomo-report-20260510T020000Z.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    assert latest_report_path(tmp_path) == newer
