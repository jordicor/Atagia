"""Tests for the in-process storage backend."""

import asyncio

import pytest

from atagia.core.storage_backend import InProcessBackend


@pytest.mark.asyncio
async def test_recent_window_round_trip_uses_copies() -> None:
    backend = InProcessBackend()
    messages = [{"id": "msg_1", "text": "hello"}]

    await backend.set_recent_window("conversation:1", messages)
    fetched = await backend.get_recent_window("conversation:1")

    assert fetched == messages
    assert fetched is not messages
    fetched[0]["text"] = "changed"
    assert (await backend.get_recent_window("conversation:1")) == messages


@pytest.mark.asyncio
async def test_context_view_ttl_dedupe_and_locking() -> None:
    backend = InProcessBackend()

    await backend.set_context_view("ctx:1", {"items": ["one"]}, ttl_seconds=1)
    assert await backend.get_context_view("ctx:1") == {"items": ["one"]}
    await asyncio.sleep(1.05)
    assert await backend.get_context_view("ctx:1") is None

    assert await backend.remember_dedupe("dedupe:1", ttl_seconds=1) is True
    assert await backend.remember_dedupe("dedupe:1", ttl_seconds=1) is False
    await asyncio.sleep(1.05)
    assert await backend.remember_dedupe("dedupe:1", ttl_seconds=1) is True

    first_lock = await backend.acquire_lock("lock:1", ttl_seconds=1)
    assert first_lock is not None
    assert await backend.acquire_lock("lock:1", ttl_seconds=1) is None
    await backend.release_lock("lock:1", "wrong-token")
    assert await backend.acquire_lock("lock:1", ttl_seconds=1) is None
    await backend.release_lock("lock:1", first_lock)
    assert await backend.acquire_lock("lock:1", ttl_seconds=1) is not None


@pytest.mark.asyncio
async def test_delete_context_view_removes_single_entry() -> None:
    backend = InProcessBackend()

    await backend.set_context_view(
        "ctx:1",
        {"user_id": "usr_1", "items": ["one"]},
        ttl_seconds=10,
    )
    await backend.set_context_view(
        "ctx:2",
        {"user_id": "usr_1", "items": ["two"]},
        ttl_seconds=10,
    )

    await backend.delete_context_view("ctx:1")

    assert await backend.get_context_view("ctx:1") is None
    assert await backend.get_context_view("ctx:2") == {"user_id": "usr_1", "items": ["two"]}


@pytest.mark.asyncio
async def test_delete_context_views_for_user_wipes_only_indexed_entries() -> None:
    backend = InProcessBackend()

    await backend.set_context_view(
        "ctx:1",
        {"user_id": "usr_1", "items": ["one"]},
        ttl_seconds=10,
    )
    await backend.set_context_view(
        "ctx:2",
        {"user_id": "usr_1", "items": ["two"]},
        ttl_seconds=10,
    )
    await backend.set_context_view(
        "ctx:3",
        {"user_id": "usr_2", "items": ["three"]},
        ttl_seconds=10,
    )
    await backend.set_context_view(
        "ctx:legacy",
        {"items": ["legacy"]},
        ttl_seconds=10,
    )

    deleted = await backend.delete_context_views_for_user("usr_1")

    assert deleted == 2
    assert await backend.get_context_view("ctx:1") is None
    assert await backend.get_context_view("ctx:2") is None
    assert await backend.get_context_view("ctx:3") == {"user_id": "usr_2", "items": ["three"]}
    assert await backend.get_context_view("ctx:legacy") == {"items": ["legacy"]}


@pytest.mark.asyncio
async def test_set_context_view_if_newer_rejects_older_publish() -> None:
    backend = InProcessBackend()

    first = await backend.set_context_view_if_newer(
        "ctx:1",
        {"user_id": "usr_1", "value": "older"},
        ttl_seconds=10,
        monotonic_seq=3,
    )
    second = await backend.set_context_view_if_newer(
        "ctx:1",
        {"user_id": "usr_1", "value": "stale"},
        ttl_seconds=10,
        monotonic_seq=2,
    )
    third = await backend.set_context_view_if_newer(
        "ctx:1",
        {"user_id": "usr_1", "value": "newer"},
        ttl_seconds=10,
        monotonic_seq=4,
    )

    assert first is True
    assert second is False
    assert third is True
    assert await backend.get_context_view("ctx:1") == {"user_id": "usr_1", "value": "newer"}


@pytest.mark.asyncio
async def test_job_queue_round_trip_and_timeout() -> None:
    backend = InProcessBackend()

    assert await backend.dequeue_job("ingest", timeout_seconds=0) is None
    await backend.enqueue_job("ingest", {"job_id": "job_1"})
    await backend.enqueue_job("ingest", {"job_id": "job_2"})

    assert await backend.dequeue_job("ingest", timeout_seconds=0) == {"job_id": "job_1"}
    assert await backend.dequeue_job("ingest", timeout_seconds=0.1) == {"job_id": "job_2"}
    assert await backend.dequeue_job("ingest", timeout_seconds=0.05) is None


@pytest.mark.asyncio
async def test_stream_pending_messages_can_be_reclaimed_before_ack() -> None:
    backend = InProcessBackend()
    await backend.stream_ensure_group("atagia:test", "workers")
    message_id = await backend.stream_add("atagia:test", {"job_id": "job_1"})

    first_read = await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )
    second_read = await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )
    reclaimed = await backend.stream_claim_idle(
        "atagia:test",
        "workers",
        "consumer-2",
        min_idle_ms=0,
        count=1,
    )

    assert [message.message_id for message in first_read] == [message_id]
    assert second_read == []
    assert [message.message_id for message in reclaimed] == [message_id]
    assert reclaimed[0].delivery_count == 2

    await backend.stream_ack("atagia:test", "workers", message_id)
    assert await backend.stream_claim_idle(
        "atagia:test",
        "workers",
        "consumer-2",
        min_idle_ms=0,
        count=1,
    ) == []


@pytest.mark.asyncio
async def test_stream_drain_waits_for_pending_ack() -> None:
    backend = InProcessBackend()
    await backend.stream_ensure_group("atagia:test", "workers")
    message_id = await backend.stream_add("atagia:test", {"job_id": "job_1"})
    messages = await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )

    assert [message.message_id for message in messages] == [message_id]

    async def _ack_later() -> None:
        await asyncio.sleep(0.05)
        await backend.stream_ack("atagia:test", "workers", message_id)

    ack_task = asyncio.create_task(_ack_later())
    try:
        assert await backend.drain(timeout_seconds=0.5) is True
    finally:
        await ack_task


@pytest.mark.asyncio
async def test_stream_drain_times_out_when_pending_work_remains() -> None:
    backend = InProcessBackend()
    await backend.stream_ensure_group("atagia:test", "workers")
    await backend.stream_add("atagia:test", {"job_id": "job_1"})
    await backend.stream_read(
        "atagia:test",
        "workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )

    assert await backend.drain(timeout_seconds=0.05) is False
