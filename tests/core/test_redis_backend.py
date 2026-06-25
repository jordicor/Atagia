"""Tests for Redis-backed storage backend behavior without a real Redis server."""

from __future__ import annotations

import asyncio
import fnmatch
import json
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

    async def zcard(self, key: str) -> int:
        del key
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


class FakeRedisPurgeClient:
    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {
            "queue:dead_letter:atagia:extract": [
                json.dumps({"payload": {"user_id": "usr_1", "conversation_id": "cnv_1"}}),
                json.dumps({"payload": {"user_id": "usr_2", "conversation_id": "cnv_2"}}),
            ]
        }
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {
            "atagia:extract": [
                ("1-0", {"payload": json.dumps({"user_id": "usr_1", "conversation_id": "cnv_1"})}),
                ("2-0", {"payload": json.dumps({"user_id": "usr_2", "conversation_id": "cnv_2"})}),
            ]
        }
        self.acked: list[tuple[str, str, str]] = []
        self.deleted: list[tuple[str, str]] = []
        self._sorted_sets: dict[str, dict[str, float]] = {}

    async def scan_iter(self, match: str = "*"):
        for key in [*self._lists, *self._streams]:
            if fnmatch.fnmatch(key, match):
                yield key

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        del start, end
        return list(self._lists.get(key, []))

    async def delete(self, key: str) -> int:
        if key in self._lists:
            self._lists.pop(key, None)
            return 1
        return 0

    async def rpush(self, key: str, *values: str) -> int:
        self._lists.setdefault(key, []).extend(values)
        return len(self._lists[key])

    async def type(self, key: str) -> str:
        if key in self._streams:
            return "stream"
        if key in self._lists:
            return "list"
        return "none"

    async def xrange(self, key: str, start: str, end: str) -> list[tuple[str, dict[str, str]]]:
        del start, end
        return list(self._streams.get(key, []))

    async def xinfo_groups(self, stream_name: str) -> list[dict[str, object]]:
        del stream_name
        return [{"name": "atagia-workers"}]

    async def xack(self, stream_name: str, group_name: str, message_id: str) -> int:
        self.acked.append((stream_name, group_name, message_id))
        return 1

    async def xdel(self, stream_name: str, message_id: str) -> int:
        self.deleted.append((stream_name, message_id))
        entries = self._streams.get(stream_name, [])
        self._streams[stream_name] = [
            entry for entry in entries if entry[0] != message_id
        ]
        return 1

    async def zrange(self, key: str, start: int, end: int) -> list[str]:
        del start, end
        return list(self._sorted_sets.get(key, {}))

    async def zrem(self, key: str, member: str) -> int:
        members = self._sorted_sets.get(key)
        if members is None or member not in members:
            return 0
        members.pop(member, None)
        return 1


class FakeRedisDeferredClient:
    def __init__(self, *, fail_promotion: bool = False) -> None:
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._sorted_sets: dict[str, dict[str, float]] = {}
        self._groups: set[tuple[str, str]] = set()
        self._next_stream_id = 1
        self.eval_calls: list[tuple[str, int, tuple[object, ...]]] = []
        self.fail_promotion = fail_promotion

    async def xgroup_create(
        self,
        *,
        name: str,
        groupname: str,
        id: str,
        mkstream: bool,
    ) -> None:
        del id, mkstream
        self._groups.add((name, groupname))
        self._streams.setdefault(name, [])

    async def eval(self, script: str, numkeys: int, *args: object) -> int:
        self.eval_calls.append((script, numkeys, args))
        keys = [str(value) for value in args[:numkeys]]
        argv = [str(value) for value in args[numkeys:]]
        script_lower = script.lower()
        if "xack" in script_lower and "zadd" in script_lower:
            deferred_key = keys[1]
            member = argv[2]
            score = float(argv[3])
            self._sorted_sets.setdefault(deferred_key, {})[member] = score
            return 1
        if "zrangebyscore" in script_lower and "xadd" in script_lower:
            if self.fail_promotion:
                raise RuntimeError("xadd failed")
            deferred_key, stream_name = keys
            now = float(argv[0])
            limit = int(argv[1])
            candidates = [
                (score, member)
                for member, score in self._sorted_sets.get(deferred_key, {}).items()
                if score <= now
            ]
            candidates.sort(key=lambda item: item[0])
            promoted = 0
            for _score, member in candidates[:limit]:
                decoded = json.loads(member)
                payload_json = decoded.get("payload_json")
                if not isinstance(payload_json, str):
                    self._sorted_sets.get(deferred_key, {}).pop(member, None)
                    continue
                message_id = f"{self._next_stream_id}-0"
                self._next_stream_id += 1
                self._streams.setdefault(stream_name, []).append(
                    (
                        message_id,
                        {"payload": payload_json},
                    )
                )
                self._sorted_sets.get(deferred_key, {}).pop(member, None)
                promoted += 1
            return promoted
        raise AssertionError("Unexpected Redis eval script")

    async def xreadgroup(
        self,
        *,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int,
        block: int | None,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        del groupname, consumername, block
        stream_name = next(iter(streams))
        entries = self._streams.get(stream_name, [])[:count]
        self._streams[stream_name] = self._streams.get(stream_name, [])[count:]
        return [(stream_name, entries)] if entries else []

    async def zcard(self, key: str) -> int:
        return len(self._sorted_sets.get(key, {}))

    async def zrange(self, key: str, start: int, end: int) -> list[str]:
        del start, end
        return list(self._sorted_sets.get(key, {}))

    async def zrem(self, key: str, member: str) -> int:
        members = self._sorted_sets.get(key)
        if members is None or member not in members:
            return 0
        members.pop(member, None)
        return 1


def _drain_backend(states: list[tuple[int, int]]) -> RedisBackend:
    backend = object.__new__(RedisBackend)
    backend._client = FakeRedisDrainClient(states)
    backend._stream_groups = {("atagia:test", "atagia-workers")}
    backend._stream_add_counts = {}
    backend._stream_read_counts = {}
    backend._stream_claim_counts = {}
    backend._stream_ack_counts = {}
    return backend


def _cache_backend() -> RedisBackend:
    backend = object.__new__(RedisBackend)
    backend._client = FakeRedisCacheClient()
    backend._stream_groups = set()
    return backend


def _purge_backend() -> RedisBackend:
    backend = object.__new__(RedisBackend)
    backend._client = FakeRedisPurgeClient()
    backend._stream_groups = set()
    return backend


def _deferred_backend(*, fail_promotion: bool = False) -> RedisBackend:
    backend = object.__new__(RedisBackend)
    backend._client = FakeRedisDeferredClient(fail_promotion=fail_promotion)
    backend._stream_groups = set()
    backend._stream_add_counts = {}
    backend._stream_read_counts = {}
    backend._stream_claim_counts = {}
    backend._stream_ack_counts = {}
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
async def test_redis_backend_purge_user_jobs_scans_persisted_streams_without_registered_groups() -> None:
    backend = _purge_backend()

    purged = await backend.purge_user_jobs("usr_1")

    assert purged == 2
    assert backend._client._lists["queue:dead_letter:atagia:extract"] == [
        json.dumps({"payload": {"user_id": "usr_2", "conversation_id": "cnv_2"}})
    ]
    assert [message_id for message_id, _fields in backend._client._streams["atagia:extract"]] == ["2-0"]
    assert backend._client.acked == [("atagia:extract", "atagia-workers", "1-0")]
    assert backend._client.deleted == [("atagia:extract", "1-0")]


@pytest.mark.asyncio
async def test_redis_backend_stream_defer_promotes_due_message_atomically() -> None:
    backend = _deferred_backend()

    await backend.stream_defer(
        "atagia:test",
        "atagia-workers",
        "1-0",
        {
            "job_id": "job_1",
            "recent_messages": [],
            "source_message_ids": [],
        },
        delay_seconds=0,
    )

    assert await backend._client.zcard("stream_deferred:atagia:test") == 1
    assert backend._stream_ack_counts == {"atagia:test": 1}

    messages = await backend.stream_read(
        "atagia:test",
        "atagia-workers",
        "consumer-1",
        count=1,
        block_ms=0,
    )

    assert [message.payload for message in messages] == [
        {
            "job_id": "job_1",
            "recent_messages": [],
            "source_message_ids": [],
        }
    ]
    assert await backend._client.zcard("stream_deferred:atagia:test") == 0
    assert backend._stream_add_counts == {"atagia:test": 1}
    assert len(backend._client.eval_calls) == 2


@pytest.mark.asyncio
async def test_redis_backend_stream_defer_keeps_due_message_when_promotion_fails() -> None:
    backend = _deferred_backend(fail_promotion=True)

    await backend.stream_defer(
        "atagia:test",
        "atagia-workers",
        "1-0",
        {"job_id": "job_1"},
        delay_seconds=0,
    )

    with pytest.raises(RuntimeError, match="xadd failed"):
        await backend.stream_read(
            "atagia:test",
            "atagia-workers",
            "consumer-1",
            count=1,
            block_ms=0,
        )

    assert await backend._client.zcard("stream_deferred:atagia:test") == 1
    assert backend._stream_add_counts == {}


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
