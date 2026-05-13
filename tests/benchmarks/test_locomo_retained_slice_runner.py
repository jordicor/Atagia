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
                        "name": "conv-26",
                        "output": "benchmarks/results/slice/conv-26",
                        "conversation": "conv-26",
                        "questions": ["conv-26:q20", "conv-26:q24"],
                        "reuse_db": "docs/tmp/dbs/conv-26",
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
    assert job.name == "conv-26"
    assert job.conversations == ["conv-26"]
    assert job.questions == ["conv-26:q20", "conv-26:q24"]
    assert job.reuse_db == Path("docs/tmp/dbs/conv-26")
    assert job.extra_args == ["--trusted-evaluation"]


def test_build_locomo_command_uses_reuse_db_evaluate_only() -> None:
    job = RetainedLoCoMoJob(
        name="typed-conv-30",
        output=Path("out/typed/conv-30"),
        conversations=["conv-30"],
        questions=["conv-30:q24"],
        reuse_db=Path("dbs/conv-30"),
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
        "out/typed/conv-30",
        "--conversations",
        "conv-30",
        "--questions",
        "conv-30:q24",
        "--reuse-db",
        "dbs/conv-30",
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
