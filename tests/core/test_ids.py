"""Tests for prefixed identifier helpers."""

from atagia.core.ids import (
    generate_prefixed_id,
    new_belief_id,
    new_job_id,
    new_memory_id,
    new_retrieval_id,
)


def test_generate_prefixed_id_uses_requested_prefix() -> None:
    identifier = generate_prefixed_id("mem")

    assert identifier.startswith("mem_")
    assert len(identifier) > len("mem_")


def test_generate_prefixed_id_rejects_invalid_prefix() -> None:
    try:
        generate_prefixed_id("bad-prefix")
    except ValueError as exc:
        assert "Invalid identifier prefix" in str(exc)
    else:  # pragma: no cover - defensive branch.
        raise AssertionError("Expected ValueError for invalid prefix")


def test_named_helpers_generate_expected_prefixes() -> None:
    assert new_memory_id().startswith("mem_")
    assert new_retrieval_id().startswith("ret_")
    assert new_job_id().startswith("job_")
    assert new_belief_id().startswith("blf_")

