"""Tests for the no-op embedding backend."""

import pytest

from atagia.services.embeddings import NoneBackend


@pytest.mark.asyncio
async def test_none_backend_is_a_no_op() -> None:
    backend = NoneBackend()

    await backend.upsert("mem_1", "text", {"scope": "conversation"})
    assert await backend.search("query", "usr_1", top_k=5) == []
    await backend.delete("mem_1")
    assert backend.vector_limit == 0

