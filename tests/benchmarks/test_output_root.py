"""Tests for the benchmark output guardrail (`benchmarks/output_root.py`)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from benchmarks import output_root


def test_default_root_is_sibling_outside_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(output_root.ENV_OUTPUT_ROOT, raising=False)
    root = output_root.bench_output_root()
    assert root == output_root.repo_root().parent / "atagia-benchmarks"
    assert not output_root.is_inside_repo(root)


def test_env_override_is_honored(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(output_root.ENV_OUTPUT_ROOT, str(tmp_path / "external"))
    assert output_root.bench_output_root() == (tmp_path / "external").resolve()


def test_env_override_inside_repo_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    inside = output_root.repo_root() / "benchmarks" / "results"
    monkeypatch.setenv(output_root.ENV_OUTPUT_ROOT, str(inside))
    with pytest.raises(ValueError, match="inside the Atagia repo"):
        output_root.bench_output_root()


def test_is_inside_repo_detects_repo_paths() -> None:
    repo = output_root.repo_root()
    assert output_root.is_inside_repo(repo)
    assert output_root.is_inside_repo(repo / "benchmarks" / "results" / "run")
    assert not output_root.is_inside_repo(repo.parent / "atagia-benchmarks")
    # Sibling sharing a name prefix must not be flagged as inside.
    assert not output_root.is_inside_repo(repo.parent / (repo.name + "-benchmarks"))


def test_assert_outside_repo_raises_inside_and_passes_outside(tmp_path) -> None:
    with pytest.raises(ValueError):
        output_root.assert_outside_repo(output_root.repo_root() / "benchmarks" / "x")
    out = output_root.assert_outside_repo(tmp_path / "y")
    assert out == (tmp_path / "y").resolve()


def test_utc_run_id_formats_and_normalizes_tz() -> None:
    aware = datetime(2026, 6, 13, 17, 0, 0, tzinfo=timezone.utc)
    assert output_root.utc_run_id(aware) == "20260613T170000Z"
    # Naive datetimes are treated as UTC.
    naive = datetime(2026, 6, 13, 17, 0, 0)
    assert output_root.utc_run_id(naive) == "20260613T170000Z"
    # Offset datetimes are converted to UTC.
    offset = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert output_root.utc_run_id(offset) == "20260613T170000Z"


def test_has_utc_component() -> None:
    assert output_root.has_utc_component("/a/b/20260613T170000Z")
    assert output_root.has_utc_component("/a/b/20260613")
    assert not output_root.has_utc_component("/a/b/run")
    # Stamp embedded in a longer filename is not a whole-component stamp.
    assert not output_root.has_utc_component("/a/b/report-20260613T170000Z.json")


def test_resolve_output_dir_auto_stamps(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(output_root.ENV_OUTPUT_ROOT, str(tmp_path))
    out = output_root.resolve_output_dir("locomo", timestamp="20260613T170000Z")
    assert out == (tmp_path / "locomo" / "20260613T170000Z").resolve()


def test_resolve_output_dir_does_not_double_stamp(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(output_root.ENV_OUTPUT_ROOT, str(tmp_path))
    out = output_root.resolve_output_dir("locomo", "20260101T000000Z")
    assert out == (tmp_path / "locomo" / "20260101T000000Z").resolve()


def test_resolve_output_dir_rejects_inside_repo() -> None:
    with pytest.raises(ValueError):
        output_root.resolve_output_dir(
            "results", root=output_root.repo_root() / "benchmarks", timestamp="20260613T170000Z"
        )


def test_resolve_output_dir_absolute_part_is_boundary_checked(tmp_path) -> None:
    # An absolute part overrides the root but is still required to be outside.
    out = output_root.resolve_output_dir(
        tmp_path / "explicit", timestamp="20260613T170000Z"
    )
    assert out == (tmp_path / "explicit" / "20260613T170000Z").resolve()
    with pytest.raises(ValueError):
        output_root.resolve_output_dir(output_root.repo_root() / "benchmarks" / "inside")
