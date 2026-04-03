"""Tests for Redis-backed storage backend behavior without a real Redis server."""

from __future__ import annotations

import asyncio
from time import monotonic

import pytest

from atagia.core.redis_client import RedisBackend


class FakeRedisDrainClient:
    def __init__(self, states: list[tuple[int, int]]) -> None:
        self._states = list(states)
        self._last_state = states[-1] if states else (0, 0)

    async def xinfo_groups(self, stream_name: str) -> list[dict[str, object]]:
        del stream_name
        if self._states:
            self._last_state = self._states.pop(0)
        pending, lag = self._last_state
        return [{"name": "atagia-workers", "pending": pending, "lag": lag}]

    async def xlen(self, stream_name: str) -> int:
        del stream_name
        return 0

    async def aclose(self) -> None:
        return None


class FakeRedisCacheClient:
    def __init__(self) -> None:
        self._values: dict[str, tuple[str, float | None]] = {}
        self._sets: dict[str, set[str]] = {}
        self._set_expires: dict[str, float] = {}

    def _purge_expired(self) -> None:
        now = monotonic()
        expired = [
            key
            for key, (_value, expires_at) in self._values.items()
            if expires_at is not None and expires_at <= now
        ]
        for key in expired:
            self._values.pop(key, None)
        expired_sets = [
            key
            for key, expires_at in self._set_expires.items()
            if expires_at <= now
        ]
        for key in expired_sets:
            self._sets.pop(key, None)
            self._set_expires.pop(key, None)

    async def get(self, key: str) -> str | None:
        self._purge_expired()
        entry = self._values.get(key)
        if entry is None:
            return None
        return entry[0]

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        self._purge_expired()
        if nx and key in self._values:
            return False
        expires_at = None if ex is None else monotonic() + ex
        self._values[key] = (value, expires_at)
        return True

    async def delete(self, *keys: str) -> int:
        self._purge_expired()
        deleted = 0
        for key in keys:
            if key in self._values:
                self._values.pop(key, None)
                deleted += 1
            elif key in self._sets:
                self._sets.pop(key, None)
                self._set_expires.pop(key, None)
                deleted += 1
        return deleted

    async def sadd(self, key: str, member: str) -> int:
        self._purge_expired()
        members = self._sets.setdefault(key, set())
        before = len(members)
        members.add(member)
        return 1 if len(members) != before else 0

    async def smembers(self, key: str) -> set[str]:
        self._purge_expired()
        return set(self._sets.get(key, set()))

    async def srem(self, key: str, member: str) -> int:
        self._purge_expired()
        members = self._sets.get(key)
        if members is None or member not in members:
            return 0
        members.remove(member)
        if not members:
            self._sets.pop(key, None)
            self._set_expires.pop(key, None)
        return 1

    async def expire(self, key: str, seconds: int) -> bool:
        self._purge_expired()
        expires_at = monotonic() + seconds
        if key in self._values:
            value, _ = self._values[key]
            self._values[key] = (value, expires_at)
            return True
        if key in self._sets:
            self._set_expires[key] = expires_at
            return True
        return False

    async def ttl(self, key: str) -> int:
        self._purge_expired()
        if key in self._values:
            expires_at = self._values[key][1]
        elif key in self._sets:
            expires_at = self._set_expires.get(key)
        else:
            return -2
        if expires_at is None:
            return -1
        return max(0, int(expires_at - monotonic()))

    async def eval(self, script: str, numkeys: int, *args: object) -> int:
        del script
        keys = [str(value) for value in args[:numkeys]]
        argv = [str(value) for value in args[numkeys:]]
        payload = argv[0]
        ttl_seconds = int(argv[1])
        monotonic_seq = int(argv[2])
        user_id = argv[3]
        cache_key = argv[4]
        user_index_prefix = argv[5]
        current_seq = await self.get(keys[1])
        if current_seq is not None and int(current_seq) >= monotonic_seq:
            return 0

        old_owner = await self.get(keys[2])
        await self.set(keys[0], payload, ex=ttl_seconds)
        await self.set(keys[1], str(monotonic_seq), ex=ttl_seconds)

        if old_owner and old_owner != user_id:
            await self.srem(f"{user_index_prefix}{old_owner}", cache_key)

        if user_id:
            await self.set(keys[2], user_id, ex=ttl_seconds)
            await self.sadd(keys[3], cache_key)
            current_ttl = await self.ttl(keys[3])
            if current_ttl < 0 or current_ttl < ttl_seconds:
                await self.expire(keys[3], ttl_seconds)
        else:
            await self.delete(keys[2])
        return 1

    async def aclose(self) -> None:
        return None


def _drain_backend(states: list[tuple[int, int]]) -> RedisBackend:
    backend = object.__new__(RedisBackend)
    backend._client = FakeRedisDrainClient(states)
    backend._stream_groups = {("atagia:test", "atagia-workers")}
    return backend


def _cache_backend() -> RedisBackend:
    backend = object.__new__(RedisBackend)
    backend._client = FakeRedisCacheClient()
    backend._stream_groups = set()
    return backend


@pytest.mark.asyncio
async def test_redis_backend_context_view_ttl_round_trip() -> None:
    backend = _cache_backend()

    await backend.set_context_view(
        "ctx:1",
        {"user_id": "usr_1", "items": ["one"]},
        ttl_seconds=1,
    )
    assert await backend.get_context_view("ctx:1") == {"user_id": "usr_1", "items": ["one"]}
    assert await backend._client.smembers("context_view_user:usr_1") == {"ctx:1"}

    await asyncio.sleep(1.05)

    assert await backend.get_context_view("ctx:1") is None
    assert await backend._client.smembers("context_view_user:usr_1") == set()


@pytest.mark.asyncio
async def test_redis_backend_delete_context_view_and_user_wipe() -> None:
    backend = _cache_backend()

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

    await backend.delete_context_view("ctx:1")
    deleted = await backend.delete_context_views_for_user("usr_1")

    assert await backend.get_context_view("ctx:1") is None
    assert deleted == 1
    assert await backend.get_context_view("ctx:2") is None
    assert await backend.get_context_view("ctx:3") == {"user_id": "usr_2", "items": ["three"]}


@pytest.mark.asyncio
async def test_redis_backend_user_index_ttl_does_not_shrink_for_plain_writes() -> None:
    backend = _cache_backend()

    await backend.set_context_view(
        "ctx:long",
        {"user_id": "usr_1", "items": ["long"]},
        ttl_seconds=5,
    )
    await backend.set_context_view(
        "ctx:short",
        {"user_id": "usr_1", "items": ["short"]},
        ttl_seconds=1,
    )

    await asyncio.sleep(1.05)

    deleted = await backend.delete_context_views_for_user("usr_1")

    assert deleted == 1
    assert await backend.get_context_view("ctx:long") is None
    assert await backend.get_context_view("ctx:short") is None


@pytest.mark.asyncio
async def test_redis_backend_monotonic_publish_rejects_older_write() -> None:
    backend = _cache_backend()

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
async def test_redis_backend_user_index_ttl_does_not_shrink_for_monotonic_writes() -> None:
    backend = _cache_backend()

    await backend.set_context_view_if_newer(
        "ctx:long",
        {"user_id": "usr_1", "value": "long"},
        ttl_seconds=5,
        monotonic_seq=1,
    )
    await backend.set_context_view_if_newer(
        "ctx:short",
        {"user_id": "usr_1", "value": "short"},
        ttl_seconds=1,
        monotonic_seq=1,
    )

    await asyncio.sleep(1.05)

    deleted = await backend.delete_context_views_for_user("usr_1")

    assert deleted == 1
    assert await backend.get_context_view("ctx:long") is None
    assert await backend.get_context_view("ctx:short") is None


@pytest.mark.asyncio
async def test_redis_backend_monotonic_publish_expires_user_index_members() -> None:
    backend = _cache_backend()

    published = await backend.set_context_view_if_newer(
        "ctx:1",
        {"user_id": "usr_1", "value": "cached"},
        ttl_seconds=1,
        monotonic_seq=1,
    )

    assert published is True
    assert await backend._client.smembers("context_view_user:usr_1") == {"ctx:1"}

    await asyncio.sleep(1.05)

    assert await backend.get_context_view("ctx:1") is None
    assert await backend._client.smembers("context_view_user:usr_1") == set()
    assert await backend.delete_context_views_for_user("usr_1") == 0


@pytest.mark.asyncio
async def test_redis_backend_drain_waits_for_stable_empty_window() -> None:
    backend = _drain_backend(
        [
            (1, 1),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ]
    )

    assert await backend.drain(timeout_seconds=0.5) is True


@pytest.mark.asyncio
async def test_redis_backend_drain_times_out_when_backlog_persists() -> None:
    backend = _drain_backend([(1, 1), (1, 1), (1, 1), (1, 1)])

    assert await backend.drain(timeout_seconds=0.15) is False
