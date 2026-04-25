"""Redis-backed transient storage backend."""

from __future__ import annotations

import asyncio
import json
from math import ceil
from time import monotonic
from typing import Any
from uuid import uuid4

from atagia.core.canonical import canonical_json_hash
from atagia.core.storage_backend import (
    StorageBackend,
    extract_context_view_conversation_id,
    extract_context_view_user_id,
)
from atagia.models.schemas_jobs import StreamMessage

try:
    from redis.asyncio import Redis, from_url
    from redis.exceptions import ResponseError
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    Redis = None
    from_url = None
    ResponseError = None


RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

DRAIN_STABLE_WINDOW_SECONDS = 0.2
CONTEXT_VIEW_PREFIX = "context_view:"
CONTEXT_VIEW_SEQ_PREFIX = "context_view_seq:"
CONTEXT_VIEW_OWNER_PREFIX = "context_view_owner:"
CONTEXT_VIEW_USER_INDEX_PREFIX = "context_view_user:"
CONTEXT_VIEW_CONVERSATION_OWNER_PREFIX = "context_view_conversation_owner:"
CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX = "context_view_conversation:"

SET_CONTEXT_VIEW_IF_NEWER_SCRIPT = """
local current_seq = redis.call("get", KEYS[2])
if current_seq and tonumber(current_seq) >= tonumber(ARGV[3]) then
    return 0
end

local old_owner = redis.call("get", KEYS[3])
redis.call("set", KEYS[1], ARGV[1], "EX", tonumber(ARGV[2]))
redis.call("set", KEYS[2], ARGV[3], "EX", tonumber(ARGV[2]))

if old_owner and old_owner ~= "" and old_owner ~= ARGV[4] then
    redis.call("srem", ARGV[6] .. old_owner, ARGV[5])
end

if ARGV[4] ~= "" then
    redis.call("set", KEYS[3], ARGV[4], "EX", tonumber(ARGV[2]))
    redis.call("sadd", KEYS[4], ARGV[5])
    local current_ttl = redis.call("ttl", KEYS[4])
    if current_ttl < 0 or current_ttl < tonumber(ARGV[2]) then
        redis.call("expire", KEYS[4], tonumber(ARGV[2]))
    end
else
    redis.call("del", KEYS[3])
end

return 1
"""


class RedisBackend(StorageBackend):
    """Redis implementation for caches, queues, locks, and dedupe keys."""

    def __init__(self, redis_url: str) -> None:
        if from_url is None:
            raise RuntimeError("redis dependency is not installed")
        self._client: Redis = from_url(redis_url, decode_responses=True)
        self._stream_groups: set[tuple[str, str]] = set()

    async def get_recent_window(self, key: str) -> list[dict[str, Any]] | None:
        raw = await self._client.get(f"recent_window:{key}")
        if raw is None:
            return None
        return json.loads(raw)

    async def set_recent_window(self, key: str, messages: list[dict[str, Any]]) -> None:
        await self._client.set(
            f"recent_window:{key}",
            json.dumps(messages, ensure_ascii=False, sort_keys=True),
        )

    async def get_context_view(self, key: str) -> dict[str, Any] | None:
        raw = await self._client.get(self._context_view_key(key))
        if raw is None:
            return None
        return json.loads(raw)

    async def set_context_view(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        serialized = json.dumps(context_view, ensure_ascii=False, sort_keys=True)
        owner_key = self._context_view_owner_key(key)
        previous_user_id = await self._client.get(owner_key)
        conversation_owner_key = self._context_view_conversation_owner_key(key)
        previous_conversation_subject = await self._client.get(conversation_owner_key)
        user_id = extract_context_view_user_id(context_view)
        conversation_id = extract_context_view_conversation_id(context_view)
        await self._client.set(
            self._context_view_key(key),
            serialized,
            ex=ttl_seconds,
        )
        await self._client.delete(self._context_view_seq_key(key))
        await self._sync_context_view_owner(
            key=key,
            user_id=user_id,
            previous_user_id=previous_user_id,
            ttl_seconds=ttl_seconds,
        )
        await self._sync_context_view_conversation_owner(
            key=key,
            user_id=user_id,
            conversation_id=conversation_id,
            previous_conversation_subject=previous_conversation_subject,
            ttl_seconds=ttl_seconds,
        )

    async def set_context_view_if_newer(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
        monotonic_seq: int,
    ) -> bool:
        user_id = extract_context_view_user_id(context_view) or ""
        conversation_id = extract_context_view_conversation_id(context_view)
        previous_conversation_subject = await self._client.get(
            self._context_view_conversation_owner_key(key)
        )
        result = await self._client.eval(
            SET_CONTEXT_VIEW_IF_NEWER_SCRIPT,
            4,
            self._context_view_key(key),
            self._context_view_seq_key(key),
            self._context_view_owner_key(key),
            self._context_view_user_index_key(user_id),
            json.dumps(context_view, ensure_ascii=False, sort_keys=True),
            ttl_seconds,
            monotonic_seq,
            user_id,
            key,
            CONTEXT_VIEW_USER_INDEX_PREFIX,
        )
        if not result:
            return False
        await self._sync_context_view_conversation_owner(
            key=key,
            user_id=user_id or None,
            conversation_id=conversation_id,
            previous_conversation_subject=previous_conversation_subject,
            ttl_seconds=ttl_seconds,
        )
        return True

    async def delete_context_view(self, key: str) -> None:
        owner_key = self._context_view_owner_key(key)
        user_id = await self._client.get(owner_key)
        conversation_subject = await self._client.get(
            self._context_view_conversation_owner_key(key)
        )
        await self._client.delete(
            self._context_view_key(key),
            self._context_view_seq_key(key),
            owner_key,
            self._context_view_conversation_owner_key(key),
        )
        if user_id:
            await self._client.srem(self._context_view_user_index_key(user_id), key)
        if conversation_subject:
            await self._client.srem(
                self._context_view_conversation_index_key(conversation_subject),
                key,
            )

    async def delete_context_views_for_user(self, user_id: str) -> int:
        cache_keys = await self._client.smembers(self._context_view_user_index_key(user_id))
        deleted = 0
        for cache_key in cache_keys:
            conversation_subject = await self._client.get(
                self._context_view_conversation_owner_key(str(cache_key))
            )
            removed = await self._client.delete(
                self._context_view_key(str(cache_key)),
                self._context_view_seq_key(str(cache_key)),
                self._context_view_owner_key(str(cache_key)),
                self._context_view_conversation_owner_key(str(cache_key)),
            )
            if int(removed or 0) > 0:
                deleted += 1
                if conversation_subject:
                    await self._client.srem(
                        self._context_view_conversation_index_key(conversation_subject),
                        str(cache_key),
                    )
                continue
            await self._client.srem(self._context_view_user_index_key(user_id), str(cache_key))
        await self._client.delete(self._context_view_user_index_key(user_id))
        return deleted

    async def delete_context_views_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> int:
        conversation_subject = self._conversation_subject(user_id, conversation_id)
        index_key = self._context_view_conversation_index_key(conversation_subject)
        cache_keys = await self._client.smembers(index_key)
        deleted = 0
        for cache_key in cache_keys:
            removed = await self._client.delete(
                self._context_view_key(str(cache_key)),
                self._context_view_seq_key(str(cache_key)),
                self._context_view_owner_key(str(cache_key)),
                self._context_view_conversation_owner_key(str(cache_key)),
            )
            if int(removed or 0) > 0:
                deleted += 1
                await self._client.srem(self._context_view_user_index_key(user_id), str(cache_key))
                continue
            await self._client.srem(index_key, str(cache_key))
        await self._client.delete(index_key)
        return deleted

    async def enqueue_job(self, queue_name: str, payload: dict[str, Any]) -> None:
        await self._client.rpush(
            f"queue:{queue_name}",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )

    async def dequeue_job(
        self,
        queue_name: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raw_payload = await self._client.lpop(f"queue:{queue_name}")
            if raw_payload is None:
                return None
            return json.loads(raw_payload)
        # None means "block forever", matching the in-process backend semantics.
        timeout = 0 if timeout_seconds is None else max(1, ceil(timeout_seconds))
        item = await self._client.blpop(f"queue:{queue_name}", timeout=timeout)
        if item is None:
            return None
        _, raw_payload = item
        return json.loads(raw_payload)

    async def stream_add(self, stream_name: str, payload: dict[str, Any]) -> str:
        return await self._client.xadd(
            stream_name,
            {"payload": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        )

    async def stream_read(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        count: int,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        response = await self._client.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_name: ">"},
            count=count,
            block=block_ms,
        )
        messages: list[StreamMessage] = []
        for _stream, entries in response:
            for message_id, fields in entries:
                normalized_fields = self._normalize_stream_fields(fields)
                raw_payload = normalized_fields.get("payload", "{}")
                messages.append(
                    StreamMessage(
                        message_id=message_id,
                        payload=json.loads(raw_payload),
                        delivery_count=1,
                    )
                )
        return messages

    async def stream_claim_idle(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        min_idle_ms: int,
        count: int,
    ) -> list[StreamMessage]:
        response = await self._client.execute_command(
            "XAUTOCLAIM",
            stream_name,
            group_name,
            consumer_name,
            min_idle_ms,
            "0-0",
            "COUNT",
            count,
        )
        if not response or len(response) < 2:
            return []

        messages: list[StreamMessage] = []
        entries = response[1]
        if not isinstance(entries, list):
            return []
        delivery_counts = await self._pending_delivery_counts(stream_name, group_name, entries)
        for message_id, fields in entries:
            normalized_fields = self._normalize_stream_fields(fields)
            raw_payload = normalized_fields.get("payload", "{}")
            messages.append(
                StreamMessage(
                    message_id=message_id,
                    payload=json.loads(raw_payload),
                    delivery_count=delivery_counts.get(message_id, 2),
                )
            )
        return messages

    async def stream_ack(self, stream_name: str, group_name: str, message_id: str) -> None:
        await self._client.xack(stream_name, group_name, message_id)

    async def stream_ensure_group(self, stream_name: str, group_name: str) -> None:
        try:
            await self._client.xgroup_create(
                name=stream_name,
                groupname=group_name,
                id="0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._stream_groups.add((stream_name, group_name))

    async def drain(self, timeout_seconds: float = 30.0) -> bool:
        if not self._stream_groups:
            return True

        deadline = monotonic() + max(0.0, timeout_seconds)
        stable_started_at: float | None = None
        while True:
            backlog_detected = False
            for stream_name, group_name in self._stream_groups:
                pending_count, lag_count = await self._group_backlog(stream_name, group_name)
                if pending_count > 0 or lag_count > 0:
                    backlog_detected = True
                    stable_started_at = None
                    break

            if not backlog_detected:
                if stable_started_at is None:
                    stable_started_at = monotonic()
                elif monotonic() - stable_started_at >= DRAIN_STABLE_WINDOW_SECONDS:
                    return True

            if monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)

    async def remember_dedupe(self, key: str, ttl_seconds: int) -> bool:
        result = await self._client.set(
            f"dedupe:{key}",
            "1",
            ex=ttl_seconds,
            nx=True,
        )
        return bool(result)

    async def force_dedupe(self, key: str, ttl_seconds: int) -> None:
        await self._client.set(f"dedupe:{key}", "1", ex=ttl_seconds)

    async def has_dedupe(self, key: str) -> bool:
        return bool(await self._client.exists(f"dedupe:{key}"))

    async def acquire_lock(self, key: str, ttl_seconds: int) -> str | None:
        token = uuid4().hex
        result = await self._client.set(
            f"lock:{key}",
            token,
            ex=ttl_seconds,
            nx=True,
        )
        return token if result else None

    async def release_lock(self, key: str, token: str) -> None:
        await self._client.eval(RELEASE_LOCK_SCRIPT, 1, f"lock:{key}", token)

    async def get_cache_generation(self, key: str) -> int:
        val = await self._client.get(f"cachegen:{key}")
        return int(val) if val else 0

    async def increment_cache_generation(self, key: str) -> int:
        return await self._client.incr(f"cachegen:{key}")

    async def close(self) -> None:
        await self._client.aclose()

    async def _sync_context_view_owner(
        self,
        *,
        key: str,
        user_id: str | None,
        previous_user_id: str | None,
        ttl_seconds: int,
    ) -> None:
        if previous_user_id and previous_user_id != user_id:
            await self._client.srem(self._context_view_user_index_key(previous_user_id), key)
        if user_id:
            await self._client.set(
                self._context_view_owner_key(key),
                user_id,
                ex=ttl_seconds,
            )
            await self._client.sadd(self._context_view_user_index_key(user_id), key)
            await self._extend_context_view_user_index_ttl(
                user_id=user_id,
                ttl_seconds=ttl_seconds,
            )
        else:
            await self._client.delete(self._context_view_owner_key(key))

    async def _sync_context_view_conversation_owner(
        self,
        *,
        key: str,
        user_id: str | None,
        conversation_id: str | None,
        previous_conversation_subject: str | None,
        ttl_seconds: int,
    ) -> None:
        conversation_subject = (
            self._conversation_subject(user_id, conversation_id)
            if user_id and conversation_id
            else None
        )
        if previous_conversation_subject and previous_conversation_subject != conversation_subject:
            await self._client.srem(
                self._context_view_conversation_index_key(previous_conversation_subject),
                key,
            )
        if conversation_subject:
            await self._client.set(
                self._context_view_conversation_owner_key(key),
                conversation_subject,
                ex=ttl_seconds,
            )
            index_key = self._context_view_conversation_index_key(conversation_subject)
            await self._client.sadd(index_key, key)
            current_ttl = await self._client.ttl(index_key)
            if current_ttl < 0 or current_ttl < ttl_seconds:
                await self._client.expire(index_key, ttl_seconds)
        else:
            await self._client.delete(self._context_view_conversation_owner_key(key))

    @staticmethod
    def _context_view_key(key: str) -> str:
        return f"{CONTEXT_VIEW_PREFIX}{key}"

    @staticmethod
    def _context_view_seq_key(key: str) -> str:
        return f"{CONTEXT_VIEW_SEQ_PREFIX}{key}"

    @staticmethod
    def _context_view_owner_key(key: str) -> str:
        return f"{CONTEXT_VIEW_OWNER_PREFIX}{key}"

    @staticmethod
    def _context_view_conversation_owner_key(key: str) -> str:
        return f"{CONTEXT_VIEW_CONVERSATION_OWNER_PREFIX}{key}"

    @staticmethod
    def _context_view_user_index_key(user_id: str) -> str:
        return f"{CONTEXT_VIEW_USER_INDEX_PREFIX}{user_id}"

    @staticmethod
    def _context_view_conversation_index_key(conversation_subject: str) -> str:
        return f"{CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX}{conversation_subject}"

    @staticmethod
    def _conversation_subject(user_id: str, conversation_id: str) -> str:
        return canonical_json_hash(
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
            }
        )

    async def _extend_context_view_user_index_ttl(
        self,
        *,
        user_id: str,
        ttl_seconds: int,
    ) -> None:
        index_key = self._context_view_user_index_key(user_id)
        current_ttl = await self._client.ttl(index_key)
        if current_ttl < 0 or current_ttl < ttl_seconds:
            await self._client.expire(index_key, ttl_seconds)

    async def _group_backlog(self, stream_name: str, group_name: str) -> tuple[int, int]:
        try:
            groups = await self._client.xinfo_groups(stream_name)
        except ResponseError as exc:
            if "no such key" in str(exc).lower():
                return 0, 0
            raise

        for group in groups:
            if str(group.get("name")) != group_name:
                continue
            pending = int(group.get("pending", 0) or 0)
            lag = group.get("lag")
            if lag is not None:
                return pending, int(lag)
            entries_read = group.get("entries-read") or group.get("entries_read")
            if entries_read is None:
                return pending, 0
            stream_length = int(await self._client.xlen(stream_name))
            return pending, max(0, stream_length - int(entries_read))
        return 0, 0

    async def _pending_delivery_counts(
        self,
        stream_name: str,
        group_name: str,
        entries: list[tuple[str, Any]],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        message_ids = [message_id for message_id, _ in entries]
        if not message_ids:
            return counts
        pending = await self._client.xpending_range(
            stream_name,
            group_name,
            message_ids[0],
            message_ids[-1],
            len(message_ids),
        )
        for item in pending:
            item_id = item.get("message_id") or item.get("messageid")
            deliveries = item.get("times_delivered") or item.get("times-delivered")
            if item_id is None or deliveries is None:
                continue
            counts[str(item_id)] = int(deliveries)
        return counts

    @staticmethod
    def _normalize_stream_fields(fields: Any) -> dict[str, Any]:
        if isinstance(fields, dict):
            return fields
        if isinstance(fields, list):
            return {
                str(fields[index]): fields[index + 1]
                for index in range(0, len(fields), 2)
            }
        return {}
