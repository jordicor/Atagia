from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.atagia_bench.__main__ import (
    _build_parser as _build_atagia_bench_parser,
)
from benchmarks.atagia_bench.__main__ import (
    _effective_benchmark_db_dir as _effective_atagia_benchmark_db_dir,
)
from benchmarks.locomo.__main__ import _build_parser as _build_locomo_parser
from benchmarks.locomo.__main__ import (
    _effective_benchmark_db_dir as _effective_locomo_benchmark_db_dir,
)
from benchmarks.retained_db_paths import (
    default_benchmark_db_dir,
    is_temporary_path,
    validate_retained_benchmark_db_dir,
)


def test_retained_db_helper_rejects_system_temp_paths() -> None:
    assert is_temporary_path("/tmp/atagia-bench/dbs")
    assert is_temporary_path("/private/tmp/atagia-bench/dbs")

    with pytest.raises(ValueError, match="Refusing to retain benchmark DB"):
        validate_retained_benchmark_db_dir("/tmp/atagia-bench/dbs")

    assert validate_retained_benchmark_db_dir(
        "/tmp/atagia-bench/dbs",
        allow_temp_benchmark_db_dir=True,
    ) == Path("/tmp/atagia-bench/dbs")


def test_default_benchmark_db_dir_uses_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    assert default_benchmark_db_dir(output_dir) == output_dir / "dbs"


def test_atagia_bench_keep_db_defaults_to_output_dbs(tmp_path: Path) -> None:
    output_dir = tmp_path / "atagia-run"
    args = _build_atagia_bench_parser().parse_args(
        [
            "--provider",
            "openai",
            "--keep-db",
            "--output",
            str(output_dir),
        ]
    )

    assert _effective_atagia_benchmark_db_dir(args) == str(output_dir / "dbs")


def test_atagia_bench_respects_explicit_benchmark_db_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "atagia-run"
    db_dir = tmp_path / "custom-dbs"
    args = _build_atagia_bench_parser().parse_args(
        [
            "--provider",
            "openai",
            "--keep-db",
            "--output",
            str(output_dir),
            "--benchmark-db-dir",
            str(db_dir),
        ]
    )

    assert _effective_atagia_benchmark_db_dir(args) == str(db_dir)


def test_locomo_keep_db_defaults_to_output_dbs(tmp_path: Path) -> None:
    output_dir = tmp_path / "locomo-run"
    args = _build_locomo_parser().parse_args(
        [
            "--data-path",
            "benchmarks/data/locomo10.json",
            "--provider",
            "openai",
            "--keep-db",
            "--output",
            str(output_dir),
        ]
    )

    assert _effective_locomo_benchmark_db_dir(args) == str(output_dir / "dbs")
