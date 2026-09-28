"""Redis-backed transient storage backend."""

from __future__ import annotations

import asyncio
from math import ceil
from time import monotonic
from typing import Any
from uuid import uuid4

from atagia.core import json_utils
from atagia.core.canonical import canonical_json_hash
from atagia.core.storage_backend import (
    DrainProgressCallback,
    LegacyTransientPurgeResult,
    RecentWindowIdentity,
    StorageBackend,
    StorageDrainSnapshot,
    _ATAGIA_JOB_STREAM_NAMES,
    _LEGACY_ATAGIA_LIST_QUEUE_NAMES,
    _is_legacy_extractor_dedupe_key_for_user,
    _job_matches_conversation,
    _job_matches_user,
    _is_twelve_hex_namespace,
    _lifecycle_diagnostic_metadata,
    _recent_window_identity_is_valid,
    _validated_job_lock_scope,
    _validated_lifecycle_coordinates,
    _unwrap_lifecycle_diagnostic,
    _wrap_lifecycle_diagnostic,
    build_recent_window_key,
    emit_drain_progress,
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

ACQUIRE_LIFECYCLE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) ~= ARGV[1] then
    return -1
end

local current = redis.call("get", KEYS[2])
if ARGV[4] == "" then
    if current then
        return 0
    end
    redis.call("set", KEYS[2], ARGV[2], "EX", tonumber(ARGV[3]))
else
    local incoming_fence = tonumber(ARGV[5])
    local high_water_raw = redis.call("get", KEYS[4])
    if high_water_raw then
        local high_water = tonumber(high_water_raw)
        if not high_water or high_water >= incoming_fence then
            return 0
        end
    end
    if current then
        local current_ok, current_value = pcall(cjson.decode, current)
        if not current_ok or type(current_value) ~= "table"
            or type(current_value["job_id"]) ~= "string"
            or current_value["job_id"] ~= ARGV[4]
            or type(current_value["execution_fence"]) ~= "number"
            or current_value["execution_fence"] >= incoming_fence then
            return 0
        end
    end
    redis.call(
        "set",
        KEYS[2],
        cjson.encode({
            job_id = ARGV[4],
            execution_fence = incoming_fence,
            token = ARGV[2]
        }),
        "EX",
        tonumber(ARGV[3])
    )
    redis.call("set", KEYS[4], incoming_fence, "EX", tonumber(ARGV[3]))
end

local redis_time = redis.call("time")
local now = tonumber(redis_time[1]) + tonumber(redis_time[2]) / 1000000
redis.call("zremrangebyscore", KEYS[3], "-inf", now)
redis.call("zadd", KEYS[3], now + tonumber(ARGV[3]), KEYS[2])
if ARGV[4] ~= "" then
    redis.call("zadd", KEYS[3], now + tonumber(ARGV[3]), KEYS[4])
end
local latest = redis.call("zrange", KEYS[3], -1, -1, "WITHSCORES")
if #latest == 2 then
    local index_ttl = math.max(1, math.ceil(tonumber(latest[2]) - now))
    redis.call("expire", KEYS[3], index_ttl)
end
return 1
"""

RELEASE_LIFECYCLE_LOCK_SCRIPT = """
local current = redis.call("get", KEYS[1])
if not current then
    return 0
end

local current_ok, current_value = pcall(cjson.decode, current)
if current_ok and type(current_value) == "table"
    and type(current_value["job_id"]) == "string"
    and type(current_value["execution_fence"]) == "number"
    and type(current_value["token"]) == "string" then
    if current_value["token"] ~= ARGV[1] then
        return 0
    end
    if ARGV[2] ~= "" and (
        current_value["job_id"] ~= ARGV[2]
        or current_value["execution_fence"] ~= tonumber(ARGV[3])
    ) then
        return 0
    end
    redis.call("del", KEYS[1])
    redis.call("zrem", KEYS[2], KEYS[1])
    if redis.call("zcard", KEYS[2]) == 0 then
        redis.call("del", KEYS[2])
    end
    return 1
end

if ARGV[2] ~= "" or current ~= ARGV[1] then
    return 0
end
redis.call("del", KEYS[1])
redis.call("zrem", KEYS[2], KEYS[1])
if redis.call("zcard", KEYS[2]) == 0 then
    redis.call("del", KEYS[2])
end
return 1
"""

DRAIN_STABLE_WINDOW_SECONDS = 0.2
ATAGIA_QUEUE_PREFIX = "atagia:queue:"
CONTEXT_VIEW_PREFIX = "context_view:"
CONTEXT_VIEW_SEQ_PREFIX = "context_view_seq:"
CONTEXT_VIEW_OWNER_PREFIX = "context_view_owner:"
CONTEXT_VIEW_USER_INDEX_PREFIX = "context_view_user:"
CONTEXT_VIEW_CONVERSATION_OWNER_PREFIX = "context_view_conversation_owner:"
CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX = "context_view_conversation:"
CONTEXT_VIEW_LIFECYCLE_OWNER_PREFIX = "context_view_lifecycle_owner:"
LIFECYCLE_MIRROR_PREFIX = "job_lifecycle:"
LIFECYCLE_DELIVERY_INDEX_PREFIX = "job_lifecycle_deliveries:"
LIFECYCLE_DELIVERY_OWNER_PREFIX = "job_delivery_owner:"
LIFECYCLE_DIAGNOSTIC_INDEX_PREFIX = "job_lifecycle_diagnostics:"
LIFECYCLE_RECENT_INDEX_PREFIX = "lifecycle_recent_entries:"
LIFECYCLE_CONTEXT_INDEX_PREFIX = "lifecycle_context_entries:"
LIFECYCLE_LOCK_PREFIX = "lifecycle_lock:"
LIFECYCLE_LOCK_HIGH_WATER_PREFIX = "lifecycle_lock_fence:"
LIFECYCLE_LOCK_INDEX_PREFIX = "lifecycle_lock_entries:"
RECENT_WINDOW_CACHE_IDENTITY_PREFIX = "recent_window_cache_identity:"
RECENT_WINDOW_USER_INDEX_PREFIX = "recent_window_user:"

SET_CONTEXT_VIEW_SCRIPT = """
local old_owner = redis.call("get", KEYS[3])
if old_owner and old_owner ~= "" and old_owner ~= ARGV[3] then
    redis.call("srem", ARGV[5] .. old_owner, ARGV[4])
end
local old_conversation = redis.call("get", KEYS[5])
if old_conversation and old_conversation ~= "" and old_conversation ~= ARGV[6] then
    redis.call("srem", ARGV[7] .. old_conversation, ARGV[4])
end
local old_lifecycle_owner = redis.call("get", KEYS[7])
if old_lifecycle_owner then
    local old_lifecycle_index = ARGV[8] .. old_lifecycle_owner
    redis.call("srem", old_lifecycle_index, "c" .. string.char(31) .. ARGV[4])
    if redis.call("scard", old_lifecycle_index) == 0 then
        redis.call("del", old_lifecycle_index)
    end
end

redis.call("set", KEYS[1], ARGV[1], "EX", tonumber(ARGV[2]))
redis.call("del", KEYS[2], KEYS[7])
if ARGV[3] ~= "" then
    redis.call("set", KEYS[3], ARGV[3], "EX", tonumber(ARGV[2]))
    redis.call("sadd", KEYS[4], ARGV[4])
    local user_ttl = redis.call("ttl", KEYS[4])
    if user_ttl < 0 or user_ttl < tonumber(ARGV[2]) then
        redis.call("expire", KEYS[4], tonumber(ARGV[2]))
    end
else
    redis.call("del", KEYS[3])
end
if ARGV[6] ~= "" then
    redis.call("set", KEYS[5], ARGV[6], "EX", tonumber(ARGV[2]))
    redis.call("sadd", KEYS[6], ARGV[4])
    local conversation_ttl = redis.call("ttl", KEYS[6])
    if conversation_ttl < 0 or conversation_ttl < tonumber(ARGV[2]) then
        redis.call("expire", KEYS[6], tonumber(ARGV[2]))
    end
else
    redis.call("del", KEYS[5])
end
return 1
"""

SET_CONTEXT_VIEW_IF_NEWER_SCRIPT = """
local current_seq = redis.call("get", KEYS[2])
if current_seq and tonumber(current_seq) >= tonumber(ARGV[3]) then
    return 0
end

local old_owner = redis.call("get", KEYS[3])
local old_lifecycle_owner = redis.call("get", KEYS[5])
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

if old_lifecycle_owner then
    local old_lifecycle_index = ARGV[7] .. old_lifecycle_owner
    redis.call("srem", old_lifecycle_index, "c" .. string.char(31) .. ARGV[5])
    if redis.call("scard", old_lifecycle_index) == 0 then
        redis.call("del", old_lifecycle_index)
    end
end
redis.call("del", KEYS[5])

return 1
"""

DELETE_CONTEXT_VIEW_SCRIPT = """
local owner = redis.call("get", KEYS[3])
local conversation_owner = redis.call("get", KEYS[4])
if ARGV[5] ~= "" and owner and owner ~= ARGV[5] then
    redis.call("srem", ARGV[2] .. ARGV[5], ARGV[1])
    return 0
end
if ARGV[6] ~= "" and conversation_owner
    and conversation_owner ~= ARGV[6] then
    redis.call("srem", ARGV[3] .. ARGV[6], ARGV[1])
    return 0
end
if owner then
    local user_index = ARGV[2] .. owner
    redis.call("srem", user_index, ARGV[1])
    if redis.call("scard", user_index) == 0 then
        redis.call("del", user_index)
    end
end
if conversation_owner then
    local conversation_index = ARGV[3] .. conversation_owner
    redis.call("srem", conversation_index, ARGV[1])
    if redis.call("scard", conversation_index) == 0 then
        redis.call("del", conversation_index)
    end
end
local lifecycle_owner = redis.call("get", KEYS[5])
if lifecycle_owner then
    local lifecycle_index = ARGV[4] .. lifecycle_owner
    redis.call("srem", lifecycle_index, "c" .. string.char(31) .. ARGV[1])
    if redis.call("scard", lifecycle_index) == 0 then
        redis.call("del", lifecycle_index)
    end
end
local existed = redis.call("exists", KEYS[1])
redis.call("del", KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5])
return existed
"""

SET_RECENT_WINDOW_FOR_LIFECYCLE_SCRIPT = """
if redis.call("get", KEYS[1]) ~= ARGV[1] then
    return 0
end

local function identity_is_complete(identity)
    if type(identity) ~= "table"
        or type(identity["user_id"]) ~= "string"
        or identity["user_id"] == ""
        or type(identity["conversation_id"]) ~= "string"
        or identity["conversation_id"] == ""
        or type(identity["lifecycle_cleanup_key"]) ~= "string"
        or identity["lifecycle_cleanup_key"] == ""
        or type(identity["lifecycle_epoch"]) ~= "string"
        or identity["lifecycle_epoch"] == ""
        or type(identity["conversation_lifecycle_epoch"]) ~= "string"
        or identity["conversation_lifecycle_epoch"] == ""
        or type(identity["cache_revision"]) ~= "number"
        or type(identity["derivation_revision"]) ~= "number"
        or type(identity["conversation_source_revision"]) ~= "number" then
        return false
    end
    local cache_revision = identity["cache_revision"]
    local derivation_revision = identity["derivation_revision"]
    local conversation_revision = identity["conversation_source_revision"]
    return cache_revision >= 0 and cache_revision == math.floor(cache_revision)
        and derivation_revision >= 0
        and derivation_revision == math.floor(derivation_revision)
        and conversation_revision >= 0
        and conversation_revision == math.floor(conversation_revision)
end

local incoming_ok, incoming = pcall(cjson.decode, ARGV[4])
if not incoming_ok or not identity_is_complete(incoming)
    or incoming["user_id"] ~= ARGV[8]
    or incoming["conversation_id"] ~= ARGV[9] then
    return 0
end
local current_raw = redis.call("get", KEYS[4])
local current = nil
if current_raw then
    local current_ok
    current_ok, current = pcall(cjson.decode, current_raw)
    if not current_ok or not identity_is_complete(current) then
        return 0
    end
    if current["user_id"] ~= incoming["user_id"]
        or current["conversation_id"] ~= incoming["conversation_id"] then
        return 0
    end
    if current["lifecycle_epoch"] ~= incoming["lifecycle_epoch"] then
        local current_mirror = redis.call(
            "get",
            ARGV[6] .. current["lifecycle_cleanup_key"]
        )
        if current_mirror == "active:" .. current["lifecycle_epoch"] then
            return 0
        end
    else
        if current["lifecycle_cleanup_key"]
            ~= incoming["lifecycle_cleanup_key"] then
            return 0
        end
        local current_cache = tonumber(current["cache_revision"])
        local incoming_cache = tonumber(incoming["cache_revision"])
        local current_derivation = tonumber(current["derivation_revision"])
        local incoming_derivation = tonumber(incoming["derivation_revision"])
        local current_conversation_revision =
            tonumber(current["conversation_source_revision"])
        local incoming_conversation_revision =
            tonumber(incoming["conversation_source_revision"])
        if incoming_cache < current_cache then
            return 0
        end
        if incoming_cache == current_cache then
            if incoming_derivation < current_derivation then
                return 0
            end
            if incoming_derivation == current_derivation then
                if current["conversation_lifecycle_epoch"]
                    ~= incoming["conversation_lifecycle_epoch"] then
                    return 0
                end
                if incoming_conversation_revision
                    < current_conversation_revision then
                    return 0
                end
            end
        end
    end
end
local member = "r" .. string.char(31) .. ARGV[3]
if current
    and current["lifecycle_cleanup_key"]
        ~= incoming["lifecycle_cleanup_key"] then
    local old_index = ARGV[5] .. current["lifecycle_cleanup_key"]
    redis.call("srem", old_index, member)
    if redis.call("scard", old_index) == 0 then
        redis.call("del", old_index)
    end
end
redis.call("set", KEYS[2], ARGV[2])
redis.call("set", KEYS[4], ARGV[4])
redis.call("sadd", KEYS[3], member)
redis.call("sadd", KEYS[5], ARGV[3])
return 1
"""

GET_RECENT_WINDOW_IF_CACHE_IDENTITY_SCRIPT = """
if redis.call("get", KEYS[2]) ~= ARGV[1] then
    return nil
end
return redis.call("get", KEYS[1])
"""

DELETE_RECENT_WINDOW_IF_CACHE_IDENTITY_SCRIPT = """
if redis.call("get", KEYS[2]) ~= ARGV[1] then
    return 0
end
redis.call("del", KEYS[1], KEYS[2])
redis.call("srem", KEYS[3], "r" .. string.char(31) .. ARGV[2])
if redis.call("scard", KEYS[3]) == 0 then
    redis.call("del", KEYS[3])
end
redis.call("srem", KEYS[4], ARGV[2])
if redis.call("scard", KEYS[4]) == 0 then
    redis.call("del", KEYS[4])
end
return 1
"""

DELETE_RECENT_WINDOWS_FOR_USER_SCRIPT = """
local function identity_is_complete(identity)
    return type(identity) == "table"
        and type(identity["user_id"]) == "string"
        and identity["user_id"] ~= ""
        and type(identity["conversation_id"]) == "string"
        and identity["conversation_id"] ~= ""
        and type(identity["lifecycle_cleanup_key"]) == "string"
        and identity["lifecycle_cleanup_key"] ~= ""
end

local logical_keys = redis.call("smembers", KEYS[1])
local deleted = 0
for _, logical_key in ipairs(logical_keys) do
    local identity_key = ARGV[2] .. logical_key
    local current_raw = redis.call("get", identity_key)
    if not current_raw then
        redis.call("srem", KEYS[1], logical_key)
    else
        local current_ok, current = pcall(cjson.decode, current_raw)
        if current_ok and identity_is_complete(current)
            and current["user_id"] == ARGV[4] then
            local lifecycle_index = ARGV[3] .. current["lifecycle_cleanup_key"]
            local member = "r" .. string.char(31) .. logical_key
            redis.call("srem", lifecycle_index, member)
            if redis.call("scard", lifecycle_index) == 0 then
                redis.call("del", lifecycle_index)
            end
            deleted = deleted + redis.call(
                "del",
                ARGV[1] .. logical_key
            )
            redis.call("del", identity_key)
            redis.call("srem", KEYS[1], logical_key)
        end
    end
end
if redis.call("scard", KEYS[1]) == 0 then
    redis.call("del", KEYS[1])
end
return deleted
"""

DELETE_RECENT_WINDOW_FOR_CONVERSATION_SCRIPT = """
local current_raw = redis.call("get", KEYS[2])
if not current_raw then
    return 0
end
local current_ok, current = pcall(cjson.decode, current_raw)
if not current_ok or type(current) ~= "table"
    or type(current["user_id"]) ~= "string"
    or type(current["conversation_id"]) ~= "string"
    or type(current["lifecycle_cleanup_key"]) ~= "string"
    or current["user_id"] ~= ARGV[1]
    or current["conversation_id"] ~= ARGV[2] then
    return 0
end
local lifecycle_index = ARGV[4] .. current["lifecycle_cleanup_key"]
local member = "r" .. string.char(31) .. ARGV[3]
redis.call("srem", lifecycle_index, member)
if redis.call("scard", lifecycle_index) == 0 then
    redis.call("del", lifecycle_index)
end
redis.call("srem", KEYS[3], ARGV[3])
if redis.call("scard", KEYS[3]) == 0 then
    redis.call("del", KEYS[3])
end
local existed = redis.call("exists", KEYS[1])
redis.call("del", KEYS[1], KEYS[2])
return existed
"""

SET_CONTEXT_VIEW_IF_NEWER_FOR_LIFECYCLE_SCRIPT = """
if redis.call("get", KEYS[7]) ~= ARGV[7] then
    return -1
end
local current_seq = redis.call("get", KEYS[2])
if current_seq and tonumber(current_seq) >= tonumber(ARGV[3]) then
    return 0
end

local old_owner = redis.call("get", KEYS[3])
if old_owner and old_owner ~= "" and old_owner ~= ARGV[4] then
    redis.call("srem", ARGV[6] .. old_owner, ARGV[5])
end
local old_conversation = redis.call("get", KEYS[5])
if old_conversation and old_conversation ~= "" and old_conversation ~= ARGV[8] then
    redis.call("srem", ARGV[9] .. old_conversation, ARGV[5])
end
local old_lifecycle_owner = redis.call("get", KEYS[9])
if old_lifecycle_owner and old_lifecycle_owner ~= ARGV[10] then
    local old_lifecycle_index = ARGV[11] .. old_lifecycle_owner
    redis.call("srem", old_lifecycle_index, "c" .. string.char(31) .. ARGV[5])
    if redis.call("scard", old_lifecycle_index) == 0 then
        redis.call("del", old_lifecycle_index)
    end
end

redis.call("set", KEYS[1], ARGV[1], "EX", tonumber(ARGV[2]))
redis.call("set", KEYS[2], ARGV[3], "EX", tonumber(ARGV[2]))
redis.call("set", KEYS[3], ARGV[4], "EX", tonumber(ARGV[2]))
redis.call("sadd", KEYS[4], ARGV[5])
local user_ttl = redis.call("ttl", KEYS[4])
if user_ttl < 0 or user_ttl < tonumber(ARGV[2]) then
    redis.call("expire", KEYS[4], tonumber(ARGV[2]))
end

if ARGV[8] ~= "" then
    redis.call("set", KEYS[5], ARGV[8], "EX", tonumber(ARGV[2]))
    redis.call("sadd", KEYS[6], ARGV[5])
    local conversation_ttl = redis.call("ttl", KEYS[6])
    if conversation_ttl < 0 or conversation_ttl < tonumber(ARGV[2]) then
        redis.call("expire", KEYS[6], tonumber(ARGV[2]))
    end
else
    redis.call("del", KEYS[5])
end
redis.call("set", KEYS[9], ARGV[10], "EX", tonumber(ARGV[2]))
redis.call("sadd", KEYS[8], "c" .. string.char(31) .. ARGV[5])
local lifecycle_ttl = redis.call("ttl", KEYS[8])
if lifecycle_ttl < 0 or lifecycle_ttl < tonumber(ARGV[2]) then
    redis.call("expire", KEYS[8], tonumber(ARGV[2]))
end
return 1
"""

PURGE_LEGACY_CONTEXT_VIEW_SCRIPT = """
local current_raw = redis.call("get", KEYS[1])
local current_lifecycle_owner = redis.call("get", KEYS[5])
if not current_raw or current_raw ~= ARGV[1] then
    return -1
end
if ARGV[2] == "0" then
    if current_lifecycle_owner then
        return -1
    end
elseif not current_lifecycle_owner or current_lifecycle_owner ~= ARGV[3] then
    return -1
end
if current_lifecycle_owner then
    local mirror = redis.call("get", KEYS[6])
    local member = "c" .. string.char(31) .. ARGV[5]
    if mirror and string.sub(mirror, 1, 7) == "active:"
        and string.len(mirror) > 7
        and redis.call("sismember", KEYS[7], member) == 1 then
        return 0
    end
    redis.call("srem", KEYS[7], member)
    if redis.call("scard", KEYS[7]) == 0 then
        redis.call("del", KEYS[7])
    end
    redis.call("del", KEYS[5])
end
if ARGV[4] == "malformed" then
    return 2
end
if ARGV[4] ~= "delete" then
    return 0
end

local owner = redis.call("get", KEYS[3])
if owner then
    local owner_index = ARGV[6] .. owner
    redis.call("srem", owner_index, ARGV[5])
    if redis.call("scard", owner_index) == 0 then
        redis.call("del", owner_index)
    end
end
local target_index = ARGV[6] .. ARGV[8]
redis.call("srem", target_index, ARGV[5])
if redis.call("scard", target_index) == 0 then
    redis.call("del", target_index)
end
local conversation_owner = redis.call("get", KEYS[4])
if conversation_owner then
    local conversation_index = ARGV[7] .. conversation_owner
    redis.call("srem", conversation_index, ARGV[5])
    if redis.call("scard", conversation_index) == 0 then
        redis.call("del", conversation_index)
    end
end
local deleted = redis.call("exists", KEYS[1])
redis.call("del", KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5])
return deleted
"""

PURGE_ORPHAN_CONTEXT_OWNER_SCRIPT = """
local current_owner = redis.call("get", KEYS[1])
if not current_owner or current_owner ~= ARGV[1] then
    return -1
end
if redis.call("exists", KEYS[2]) == 1 then
    return 0
end
local user_index = ARGV[4] .. ARGV[3]
redis.call("srem", user_index, ARGV[2])
if redis.call("scard", user_index) == 0 then
    redis.call("del", user_index)
end
local conversation_owner = redis.call("get", KEYS[4])
if conversation_owner then
    local conversation_index = ARGV[5] .. conversation_owner
    redis.call("srem", conversation_index, ARGV[2])
    if redis.call("scard", conversation_index) == 0 then
        redis.call("del", conversation_index)
    end
end
local lifecycle_owner = redis.call("get", KEYS[5])
if lifecycle_owner then
    local lifecycle_index = ARGV[6] .. lifecycle_owner
    redis.call("srem", lifecycle_index, "c" .. string.char(31) .. ARGV[2])
    if redis.call("scard", lifecycle_index) == 0 then
        redis.call("del", lifecycle_index)
    end
end
redis.call("del", KEYS[1], KEYS[3], KEYS[4], KEYS[5])
return 1
"""

PRUNE_CONTEXT_USER_INDEX_SCRIPT = """
local logical_keys = redis.call("smembers", KEYS[1])
local removed = 0
for _, logical_key in ipairs(logical_keys) do
    local value_exists = redis.call("exists", ARGV[1] .. logical_key)
    local owner = redis.call("get", ARGV[2] .. logical_key)
    if value_exists == 0 or not owner or owner ~= ARGV[3] then
        removed = removed + redis.call("srem", KEYS[1], logical_key)
    end
end
if redis.call("scard", KEYS[1]) == 0 then
    redis.call("del", KEYS[1])
end
return removed
"""

PURGE_LEGACY_RECENT_WINDOW_SCRIPT = """
local current_identity = redis.call("get", KEYS[2])
if ARGV[1] == "0" then
    if current_identity then
        return -1
    end
elseif not current_identity or current_identity ~= ARGV[2] then
    return -1
end
if ARGV[3] == "1" then
    if redis.call("get", KEYS[3]) == ARGV[4]
        and redis.call("sismember", KEYS[4], ARGV[5]) == 1 then
        return 0
    end
    redis.call("srem", KEYS[4], ARGV[5])
    if redis.call("scard", KEYS[4]) == 0 then
        redis.call("del", KEYS[4])
    end
    redis.call("srem", KEYS[5], ARGV[6])
    if redis.call("scard", KEYS[5]) == 0 then
        redis.call("del", KEYS[5])
    end
end
local deleted = redis.call("del", KEYS[1])
redis.call("del", KEYS[2])
return deleted
"""

PRUNE_RECENT_USER_MEMBER_SCRIPT = """
local current_identity = redis.call("get", KEYS[3])
if ARGV[1] == "0" then
    if current_identity then
        return -1
    end
elseif not current_identity or current_identity ~= ARGV[2] then
    return -1
end
local value_exists = redis.call("exists", KEYS[2])
local value_type = redis.call("type", KEYS[2])["ok"]
if ARGV[4] == "1" and value_exists == 1 and value_type == "string"
    and redis.call("get", KEYS[4]) == ARGV[5]
    and redis.call("sismember", KEYS[5], ARGV[6]) == 1 then
    return 0
end
redis.call("srem", KEYS[1], ARGV[7])
if redis.call("scard", KEYS[1]) == 0 then
    redis.call("del", KEYS[1])
end
if value_exists == 0 then
    redis.call("del", KEYS[3])
    if ARGV[3] == "1" then
        redis.call("srem", KEYS[5], ARGV[6])
        if redis.call("scard", KEYS[5]) == 0 then
            redis.call("del", KEYS[5])
        end
    end
end
return 1
"""

PURGE_ORPHAN_RECENT_IDENTITY_SCRIPT = """
local current_identity = redis.call("get", KEYS[1])
if not current_identity or current_identity ~= ARGV[1] then
    return -1
end
if redis.call("exists", KEYS[2]) == 1 then
    return 0
end
redis.call("del", KEYS[1])
redis.call("srem", KEYS[3], ARGV[2])
if redis.call("scard", KEYS[3]) == 0 then
    redis.call("del", KEYS[3])
end
redis.call("srem", KEYS[4], ARGV[3])
if redis.call("scard", KEYS[4]) == 0 then
    redis.call("del", KEYS[4])
end
return 1
"""

PURGE_LEGACY_STREAM_ENTRIES_SCRIPT = """
local function is_current_notification(value)
    if type(value) ~= "table" then
        return false
    end
    local allowed = {
        job_id = true,
        dispatch_token = true,
        lifecycle_epoch = true,
        lifecycle_cleanup_key = true
    }
    local count = 0
    for key, _ in pairs(value) do
        if not allowed[key] then
            return false
        end
        count = count + 1
    end
    return count == 4
        and type(value["job_id"]) == "string"
        and value["job_id"] ~= ""
        and type(value["dispatch_token"]) == "string"
        and value["dispatch_token"] ~= ""
        and type(value["lifecycle_epoch"]) == "string"
        and value["lifecycle_epoch"] ~= ""
        and type(value["lifecycle_cleanup_key"]) == "string"
        and value["lifecycle_cleanup_key"] ~= ""
end

local function resembles_current_notification(value)
    return type(value) == "table" and (
        value["dispatch_token"] ~= nil
        or value["lifecycle_epoch"] ~= nil
        or value["lifecycle_cleanup_key"] ~= nil
    )
end

local function legacy_job_user_id(value)
    if type(value) ~= "table" then
        return nil
    end
    local allowed = {
        schema_version = true,
        job_id = true,
        job_type = true,
        user_id = true,
        parent_job_id = true,
        conversation_id = true,
        message_ids = true,
        transcript_rebuild_id = true,
        maintenance_operation_id = true,
        payload = true,
        created_at = true,
        operational_profile = true
    }
    for key, _ in pairs(value) do
        if not allowed[key] then
            return nil
        end
    end
    if type(value["job_id"]) ~= "string" or value["job_id"] == ""
        or type(value["job_type"]) ~= "string" or value["job_type"] == ""
        or type(value["user_id"]) ~= "string" or value["user_id"] == ""
        or type(value["payload"]) ~= "table" then
        return nil
    end
    if value["schema_version"] ~= nil
        and (type(value["schema_version"]) ~= "number"
            or value["schema_version"] ~= 1) then
        return nil
    end
    if value["message_ids"] ~= nil
        and type(value["message_ids"]) ~= "table" then
        return nil
    end
    return value["user_id"]
end

local entries = redis.call("xrange", KEYS[1], "-", "+")
local deleted = 0
local malformed = 0
for _, entry in ipairs(entries) do
    local payload_raw = nil
    local fields = entry[2]
    if #fields == 2 and fields[1] == "payload" then
        payload_raw = fields[2]
    end
    if not payload_raw then
        malformed = malformed + 1
    else
        local ok, payload = pcall(cjson.decode, payload_raw)
        if not ok or type(payload) ~= "table" then
            malformed = malformed + 1
        elseif is_current_notification(payload) then
            local cleanup_key = payload["lifecycle_cleanup_key"]
            local lifecycle_member = KEYS[1]
                .. string.char(31) .. entry[1]
            if redis.call("get", ARGV[2] .. cleanup_key)
                    ~= "active:" .. payload["lifecycle_epoch"]
                or redis.call(
                    "sismember",
                    ARGV[3] .. cleanup_key,
                    lifecycle_member
                ) ~= 1
                or redis.call(
                    "hget",
                    ARGV[4] .. KEYS[1],
                    entry[1]
                ) ~= cleanup_key then
                malformed = malformed + 1
            end
        elseif resembles_current_notification(payload) then
            malformed = malformed + 1
        else
            local owner = legacy_job_user_id(payload)
            if not owner then
                malformed = malformed + 1
            elseif owner == ARGV[1] then
                deleted = deleted + redis.call("xdel", KEYS[1], entry[1])
            end
        end
    end
end
return {deleted, malformed}
"""

PURGE_LEGACY_DEAD_LETTER_ITEMS_SCRIPT = """
local function is_current_notification(value)
    if type(value) ~= "table" then
        return false
    end
    local allowed = {
        job_id = true,
        dispatch_token = true,
        lifecycle_epoch = true,
        lifecycle_cleanup_key = true
    }
    local count = 0
    for key, _ in pairs(value) do
        if not allowed[key] then
            return false
        end
        count = count + 1
    end
    return count == 4
        and type(value["job_id"]) == "string"
        and value["job_id"] ~= ""
        and type(value["dispatch_token"]) == "string"
        and value["dispatch_token"] ~= ""
        and type(value["lifecycle_epoch"]) == "string"
        and value["lifecycle_epoch"] ~= ""
        and type(value["lifecycle_cleanup_key"]) == "string"
        and value["lifecycle_cleanup_key"] ~= ""
end

local function is_current_diagnostic(value)
    if type(value) ~= "table" then
        return false
    end
    local outer_allowed = {
        ["_atagia_lifecycle_diagnostic"] = true,
        payload = true
    }
    local outer_count = 0
    for key, _ in pairs(value) do
        if not outer_allowed[key] then
            return false
        end
        outer_count = outer_count + 1
    end
    local metadata = value["_atagia_lifecycle_diagnostic"]
    if outer_count ~= 2 or type(value["payload"]) ~= "table"
        or type(metadata) ~= "table" then
        return false
    end
    local metadata_allowed = {
        delivery_id = true,
        lifecycle_cleanup_key = true,
        lifecycle_epoch = true
    }
    local metadata_count = 0
    for key, _ in pairs(metadata) do
        if not metadata_allowed[key] then
            return false
        end
        metadata_count = metadata_count + 1
    end
    return metadata_count == 3
        and type(metadata["delivery_id"]) == "string"
        and metadata["delivery_id"] ~= ""
        and type(metadata["lifecycle_cleanup_key"]) == "string"
        and metadata["lifecycle_cleanup_key"] ~= ""
        and type(metadata["lifecycle_epoch"]) == "string"
        and metadata["lifecycle_epoch"] ~= ""
end

local function exact_admin_payload(value, queue_name)
    if type(value) ~= "table" then
        return nil
    end
    local count = 0
    for _, _ in pairs(value) do
        count = count + 1
    end
    if queue_name == "admin_rebuild_user" then
        if count == 1 and type(value["user_id"]) == "string"
            and value["user_id"] ~= "" then
            return value
        end
        return nil
    end
    if queue_name == "admin_rebuild_conversation" then
        if count == 2 and type(value["user_id"]) == "string"
            and value["user_id"] ~= ""
            and type(value["conversation_id"]) == "string"
            and value["conversation_id"] ~= "" then
            return value
        end
        return nil
    end
    return nil
end

local function legacy_job_user_id(value)
    if type(value) ~= "table" then
        return nil
    end
    local allowed = {
        schema_version = true,
        job_id = true,
        job_type = true,
        user_id = true,
        parent_job_id = true,
        conversation_id = true,
        message_ids = true,
        transcript_rebuild_id = true,
        maintenance_operation_id = true,
        payload = true,
        created_at = true,
        operational_profile = true
    }
    for key, _ in pairs(value) do
        if not allowed[key] then
            return nil
        end
    end
    if type(value["job_id"]) ~= "string" or value["job_id"] == ""
        or type(value["job_type"]) ~= "string" or value["job_type"] == ""
        or type(value["user_id"]) ~= "string" or value["user_id"] == ""
        or type(value["payload"]) ~= "table" then
        return nil
    end
    if value["schema_version"] ~= nil
        and (type(value["schema_version"]) ~= "number"
            or value["schema_version"] ~= 1) then
        return nil
    end
    if value["message_ids"] ~= nil
        and type(value["message_ids"]) ~= "table" then
        return nil
    end
    return value["user_id"]
end

local function exact_dead_letter_payload(value)
    if type(value) ~= "table" then
        return nil
    end
    local allowed = {
        message_id = true,
        delivery_count = true,
        payload = true,
        error = true,
        error_details = true
    }
    local count = 0
    for key, _ in pairs(value) do
        if not allowed[key] then
            return nil
        end
        count = count + 1
    end
    local delivery_count = value["delivery_count"]
    if (count ~= 4 and count ~= 5)
        or type(value["message_id"]) ~= "string"
        or value["message_id"] == ""
        or type(delivery_count) ~= "number"
        or delivery_count < 1
        or delivery_count ~= math.floor(delivery_count)
        or type(value["payload"]) ~= "table"
        or not legacy_job_user_id(value["payload"])
        or type(value["error"]) ~= "string"
        or (count == 5 and type(value["error_details"]) ~= "table") then
        return nil
    end
    return value["payload"]
end

local items = redis.call("lrange", KEYS[1], 0, -1)
local retained = {}
local deleted = 0
local malformed = 0
local ttl_ms = redis.call("pttl", KEYS[1])
local queue_name = string.sub(KEYS[1], 7)
for _, raw in ipairs(items) do
    local drop = false
    local ok, decoded = pcall(cjson.decode, raw)
    if not ok or type(decoded) ~= "table" then
        malformed = malformed + 1
    elseif is_current_diagnostic(decoded) then
        local metadata = decoded["_atagia_lifecycle_diagnostic"]
        local lifecycle_index = ARGV[3] .. metadata["lifecycle_cleanup_key"]
        local member = KEYS[1] .. string.char(31) .. raw
        local mirror = redis.call(
            "get",
            ARGV[2] .. metadata["lifecycle_cleanup_key"]
        )
        local mirror_is_active = mirror
            == "active:" .. metadata["lifecycle_epoch"]
        local is_indexed = redis.call(
            "sismember",
            lifecycle_index,
            member
        ) == 1
        if mirror_is_active and not is_indexed then
            malformed = malformed + 1
        elseif not mirror_is_active then
            redis.call("srem", lifecycle_index, member)
            if redis.call("scard", lifecycle_index) == 0 then
                redis.call("del", lifecycle_index)
            end
            local candidate = decoded["payload"]
            if type(candidate["user_id"]) ~= "string"
                or candidate["user_id"] == "" then
                malformed = malformed + 1
            elseif candidate["user_id"] == ARGV[1] then
                drop = true
            end
        end
    elseif decoded["_atagia_lifecycle_diagnostic"] ~= nil then
        malformed = malformed + 1
    elseif is_current_notification(decoded) then
        malformed = malformed + 1
    else
        local candidate = exact_admin_payload(decoded, queue_name)
        if string.sub(queue_name, 1, 12) == "dead_letter:" then
            candidate = exact_dead_letter_payload(decoded)
        end
        if not candidate or type(candidate["user_id"]) ~= "string"
            or candidate["user_id"] == "" then
            malformed = malformed + 1
        elseif candidate["user_id"] == ARGV[1] then
            drop = true
        end
    end
    if drop then
        deleted = deleted + 1
    else
        table.insert(retained, raw)
    end
end
redis.call("del", KEYS[1])
for _, raw in ipairs(retained) do
    redis.call("rpush", KEYS[1], raw)
end
if #retained > 0 and ttl_ms > 0 then
    redis.call("pexpire", KEYS[1], ttl_ms)
end
return {deleted, malformed}
"""

PURGE_LEGACY_DEFERRED_ITEMS_SCRIPT = """
local function is_current_notification(value)
    if type(value) ~= "table" then
        return false
    end
    local allowed = {
        job_id = true,
        dispatch_token = true,
        lifecycle_epoch = true,
        lifecycle_cleanup_key = true
    }
    local count = 0
    for key, _ in pairs(value) do
        if not allowed[key] then
            return false
        end
        count = count + 1
    end
    return count == 4
        and type(value["job_id"]) == "string"
        and value["job_id"] ~= ""
        and type(value["dispatch_token"]) == "string"
        and value["dispatch_token"] ~= ""
        and type(value["lifecycle_epoch"]) == "string"
        and value["lifecycle_epoch"] ~= ""
        and type(value["lifecycle_cleanup_key"]) == "string"
        and value["lifecycle_cleanup_key"] ~= ""
end

local function resembles_current_notification(value)
    return type(value) == "table" and (
        value["dispatch_token"] ~= nil
        or value["lifecycle_epoch"] ~= nil
        or value["lifecycle_cleanup_key"] ~= nil
    )
end

local members = redis.call("zrange", KEYS[1], 0, -1)
local deleted = 0
local malformed = 0
for _, raw in ipairs(members) do
    local outer_ok, outer = pcall(cjson.decode, raw)
    local outer_count = 0
    local outer_shape_ok = outer_ok and type(outer) == "table"
    if outer_shape_ok then
        for key, _ in pairs(outer) do
            if key ~= "id" and key ~= "payload_json" and key ~= "payload" then
                outer_shape_ok = false
            end
            outer_count = outer_count + 1
        end
    end
    local has_payload_json = outer_shape_ok
        and outer["payload_json"] ~= nil
    local has_payload = outer_shape_ok and outer["payload"] ~= nil
    if not outer_shape_ok or outer_count ~= 2
        or type(outer["id"]) ~= "string" or outer["id"] == ""
        or has_payload_json == has_payload
        or (has_payload_json and type(outer["payload_json"]) ~= "string")
        or (has_payload and type(outer["payload"]) ~= "table") then
        malformed = malformed + 1
    else
        local payload_ok = true
        local payload = outer["payload"]
        if has_payload_json then
            payload_ok, payload = pcall(cjson.decode, outer["payload_json"])
        end
        if not payload_ok or type(payload) ~= "table" then
            malformed = malformed + 1
        elseif is_current_notification(payload) then
            malformed = malformed + 1
        elseif resembles_current_notification(payload) then
            malformed = malformed + 1
        elseif type(payload["user_id"]) ~= "string"
            or payload["user_id"] == "" then
            malformed = malformed + 1
        elseif payload["user_id"] == ARGV[1] then
            deleted = deleted + redis.call("zrem", KEYS[1], raw)
        end
    end
end
return {deleted, malformed}
"""

PURGE_LIST_JOBS_SCRIPT = """
local items = redis.call("lrange", KEYS[1], 0, -1)
local retained = {}
local purged = 0
for _, raw in ipairs(items) do
    local drop = false
    local ok, decoded = pcall(cjson.decode, raw)
    if ok and type(decoded) == "table" then
        local job = decoded
        if type(decoded["payload"]) == "table" then
            job = decoded["payload"]
        end
        if tostring(job["user_id"] or "") == ARGV[1] then
            if ARGV[2] == "" or tostring(job["conversation_id"] or "") == ARGV[2] then
                drop = true
            end
        end
    end
    if drop then
        purged = purged + 1
    else
        table.insert(retained, raw)
    end
end
redis.call("del", KEYS[1])
for _, raw in ipairs(retained) do
    redis.call("rpush", KEYS[1], raw)
end
return purged
"""

PREPARE_LIFECYCLE_MIRROR_SCRIPT = """
local current = redis.call("get", KEYS[1])
if current and string.sub(current, 1, string.len(ARGV[2])) ~= ARGV[2] then
    return current
end
redis.call("set", KEYS[1], ARGV[1])
return ARGV[1]
"""

ACTIVATE_LIFECYCLE_MIRROR_SCRIPT = """
if redis.call("get", KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call("set", KEYS[1], ARGV[2])
return 1
"""

PUBLISH_JOB_NOTIFICATION_SCRIPT = """
if redis.call("get", KEYS[1]) ~= ARGV[1] then
    return false
end
local message_id = redis.call("xadd", KEYS[3], "*", "payload", ARGV[2])
local member = KEYS[3] .. string.char(31) .. message_id
redis.call("sadd", KEYS[2], member)
redis.call("hset", KEYS[4], message_id, ARGV[3])
return message_id
"""

PUBLISH_LIFECYCLE_DIAGNOSTIC_SCRIPT = """
if redis.call("get", KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call("rpush", KEYS[3], ARGV[2])
local member = KEYS[3] .. string.char(31) .. ARGV[2]
redis.call("sadd", KEYS[2], member)
return 1
"""

ACK_INDEXED_STREAM_MESSAGE_SCRIPT = """
local acked = redis.call("xack", KEYS[1], ARGV[1], ARGV[2])
redis.call("xdel", KEYS[1], ARGV[2])
local cleanup_key = redis.call("hget", KEYS[2], ARGV[2])
if cleanup_key then
    local index_key = ARGV[3] .. cleanup_key
    local member = KEYS[1] .. string.char(31) .. ARGV[2]
    redis.call("srem", index_key, member)
    redis.call("hdel", KEYS[2], ARGV[2])
    if redis.call("scard", index_key) == 0 then
        redis.call("del", index_key)
    end
end
return acked
"""

REVOKE_LIFECYCLE_AND_PURGE_SCRIPT = """
local function identity_is_complete(identity)
    if type(identity) ~= "table"
        or type(identity["user_id"]) ~= "string"
        or identity["user_id"] == ""
        or type(identity["conversation_id"]) ~= "string"
        or identity["conversation_id"] == ""
        or type(identity["lifecycle_cleanup_key"]) ~= "string"
        or identity["lifecycle_cleanup_key"] == ""
        or type(identity["lifecycle_epoch"]) ~= "string"
        or identity["lifecycle_epoch"] == ""
        or type(identity["conversation_lifecycle_epoch"]) ~= "string"
        or identity["conversation_lifecycle_epoch"] == ""
        or type(identity["cache_revision"]) ~= "number"
        or type(identity["derivation_revision"]) ~= "number"
        or type(identity["conversation_source_revision"]) ~= "number" then
        return false
    end
    local cache_revision = identity["cache_revision"]
    local derivation_revision = identity["derivation_revision"]
    local conversation_revision = identity["conversation_source_revision"]
    return cache_revision >= 0 and cache_revision == math.floor(cache_revision)
        and derivation_revision >= 0
        and derivation_revision == math.floor(derivation_revision)
        and conversation_revision >= 0
        and conversation_revision == math.floor(conversation_revision)
end

redis.call("set", KEYS[1], ARGV[1])
local members = redis.call("smembers", KEYS[2])
local purged = 0
for _, member in ipairs(members) do
    local separator = string.find(member, string.char(31), 1, true)
    if separator then
        local stream_name = string.sub(member, 1, separator - 1)
        local message_id = string.sub(member, separator + 1)
        redis.call("xack", stream_name, ARGV[2], message_id)
        purged = purged + redis.call("xdel", stream_name, message_id)
        redis.call("hdel", ARGV[3] .. stream_name, message_id)
    end
end
redis.call("del", KEYS[2])
local recent_members = redis.call("smembers", KEYS[3])
for _, member in ipairs(recent_members) do
    local separator = string.find(member, string.char(31), 1, true)
    if separator then
        local logical_key = string.sub(member, separator + 1)
        local identity_key = ARGV[11] .. logical_key
        local current_raw = redis.call("get", identity_key)
        if current_raw then
            local current_ok, current = pcall(cjson.decode, current_raw)
            if current_ok and identity_is_complete(current)
                and current["lifecycle_epoch"] == ARGV[12]
                and current["lifecycle_cleanup_key"] == ARGV[13] then
                local user_index = ARGV[14] .. current["user_id"]
                redis.call("srem", user_index, logical_key)
                if redis.call("scard", user_index) == 0 then
                    redis.call("del", user_index)
                end
                redis.call("del", ARGV[4] .. logical_key, identity_key)
            end
        end
    end
end
redis.call("del", KEYS[3])
local context_members = redis.call("smembers", KEYS[4])
for _, member in ipairs(context_members) do
    local separator = string.find(member, string.char(31), 1, true)
    if separator then
        local logical_key = string.sub(member, separator + 1)
        local lifecycle_owner_key = ARGV[15] .. logical_key
        if redis.call("get", lifecycle_owner_key) == ARGV[13] then
            local owner_key = ARGV[7] .. logical_key
            local owner = redis.call("get", owner_key)
            if owner then
                redis.call("srem", ARGV[9] .. owner, logical_key)
            end
            local conversation_owner_key = ARGV[8] .. logical_key
            local conversation_owner = redis.call("get", conversation_owner_key)
            if conversation_owner then
                redis.call("srem", ARGV[10] .. conversation_owner, logical_key)
            end
            redis.call(
                "del",
                ARGV[5] .. logical_key,
                ARGV[6] .. logical_key,
                owner_key,
                conversation_owner_key,
                lifecycle_owner_key
            )
        end
    end
end
redis.call("del", KEYS[4])
local diagnostic_members = redis.call("smembers", KEYS[5])
for _, member in ipairs(diagnostic_members) do
    local separator = string.find(member, string.char(31), 1, true)
    if separator then
        local queue_key = string.sub(member, 1, separator - 1)
        local raw = string.sub(member, separator + 1)
        purged = purged + redis.call("lrem", queue_key, 1, raw)
    end
end
redis.call("del", KEYS[5])
local lifecycle_lock_keys = redis.call("zrange", KEYS[6], 0, -1)
for _, transient_key in ipairs(lifecycle_lock_keys) do
    redis.call("del", transient_key)
end
redis.call("del", KEYS[6])
return purged
"""


class RedisBackend(StorageBackend):
    """Redis implementation for caches, queues, locks, and dedupe keys."""

    def __init__(self, redis_url: str) -> None:
        if from_url is None:
            raise RuntimeError("redis dependency is not installed")
        self._client: Redis = from_url(redis_url, decode_responses=True)
        self._stream_groups: set[tuple[str, str]] = set()
        self._stream_add_counts: dict[str, int] = {}
        self._stream_read_counts: dict[str, int] = {}
        self._stream_claim_counts: dict[str, int] = {}
        self._stream_ack_counts: dict[str, int] = {}

    async def get_recent_window_for_cache_identity(
        self,
        key: str,
        *,
        user_id: str,
        conversation_id: str,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        cache_revision: int,
        derivation_revision: int,
        conversation_lifecycle_epoch: str,
        conversation_source_revision: int,
    ) -> list[dict[str, Any]] | None:
        identity = RecentWindowIdentity(
            user_id=user_id,
            conversation_id=conversation_id,
            lifecycle_cleanup_key=lifecycle_cleanup_key,
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
            conversation_lifecycle_epoch=conversation_lifecycle_epoch,
            conversation_source_revision=conversation_source_revision,
        )
        if not _recent_window_identity_is_valid(
            identity
        ) or key != build_recent_window_key(user_id, conversation_id):
            return None
        # One script, so the identity cannot change between the check and the
        # read: two round-trips could pass the fence against one publication and
        # return the payload of the next.
        raw = await self._client.eval(
            GET_RECENT_WINDOW_IF_CACHE_IDENTITY_SCRIPT,
            2,
            f"recent_window:{key}",
            self._recent_window_cache_identity_key(key),
            self._recent_window_cache_identity(identity),
        )
        if raw is None:
            return None
        return json_utils.loads(raw)

    async def set_recent_window_for_lifecycle(
        self,
        key: str,
        messages: list[dict[str, Any]],
        *,
        user_id: str,
        conversation_id: str,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        cache_revision: int,
        derivation_revision: int,
        conversation_lifecycle_epoch: str,
        conversation_source_revision: int,
    ) -> bool:
        identity = RecentWindowIdentity(
            user_id=user_id,
            conversation_id=conversation_id,
            lifecycle_cleanup_key=lifecycle_cleanup_key,
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
            conversation_lifecycle_epoch=conversation_lifecycle_epoch,
            conversation_source_revision=conversation_source_revision,
        )
        if not _recent_window_identity_is_valid(
            identity
        ) or key != build_recent_window_key(user_id, conversation_id):
            return False
        result = await self._client.eval(
            SET_RECENT_WINDOW_FOR_LIFECYCLE_SCRIPT,
            5,
            self._lifecycle_mirror_key(lifecycle_cleanup_key),
            f"recent_window:{key}",
            self._lifecycle_recent_index_key(lifecycle_cleanup_key),
            self._recent_window_cache_identity_key(key),
            self._recent_window_user_index_key(user_id),
            f"active:{lifecycle_epoch}",
            json_utils.dumps(messages, sort_keys=True),
            key,
            self._recent_window_cache_identity(identity),
            LIFECYCLE_RECENT_INDEX_PREFIX,
            LIFECYCLE_MIRROR_PREFIX,
            RECENT_WINDOW_USER_INDEX_PREFIX,
            user_id,
            conversation_id,
        )
        return int(result or 0) == 1

    async def delete_recent_window_if_cache_identity(
        self,
        key: str,
        *,
        user_id: str,
        conversation_id: str,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        cache_revision: int,
        derivation_revision: int,
        conversation_lifecycle_epoch: str,
        conversation_source_revision: int,
    ) -> bool:
        identity = RecentWindowIdentity(
            user_id=user_id,
            conversation_id=conversation_id,
            lifecycle_cleanup_key=lifecycle_cleanup_key,
            lifecycle_epoch=lifecycle_epoch,
            cache_revision=cache_revision,
            derivation_revision=derivation_revision,
            conversation_lifecycle_epoch=conversation_lifecycle_epoch,
            conversation_source_revision=conversation_source_revision,
        )
        if not _recent_window_identity_is_valid(
            identity
        ) or key != build_recent_window_key(user_id, conversation_id):
            return False
        result = await self._client.eval(
            DELETE_RECENT_WINDOW_IF_CACHE_IDENTITY_SCRIPT,
            4,
            f"recent_window:{key}",
            self._recent_window_cache_identity_key(key),
            self._lifecycle_recent_index_key(lifecycle_cleanup_key),
            self._recent_window_user_index_key(user_id),
            self._recent_window_cache_identity(identity),
            key,
        )
        return int(result or 0) == 1

    async def delete_recent_windows_for_user(self, user_id: str) -> int:
        return int(
            await self._client.eval(
                DELETE_RECENT_WINDOWS_FOR_USER_SCRIPT,
                1,
                self._recent_window_user_index_key(user_id),
                "recent_window:",
                RECENT_WINDOW_CACHE_IDENTITY_PREFIX,
                LIFECYCLE_RECENT_INDEX_PREFIX,
                user_id,
            )
            or 0
        )

    async def delete_recent_window_for_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> int:
        logical_key = build_recent_window_key(user_id, conversation_id)
        return int(
            await self._client.eval(
                DELETE_RECENT_WINDOW_FOR_CONVERSATION_SCRIPT,
                3,
                f"recent_window:{logical_key}",
                self._recent_window_cache_identity_key(logical_key),
                self._recent_window_user_index_key(user_id),
                user_id,
                conversation_id,
                logical_key,
                LIFECYCLE_RECENT_INDEX_PREFIX,
            )
            or 0
        )

    async def get_context_view(self, key: str) -> dict[str, Any] | None:
        raw = await self._client.get(self._context_view_key(key))
        if raw is None:
            return None
        return json_utils.loads(raw)

    async def set_context_view(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        serialized = json_utils.dumps(context_view, sort_keys=True)
        user_id = extract_context_view_user_id(context_view) or ""
        conversation_id = extract_context_view_conversation_id(context_view)
        conversation_subject = (
            self._conversation_subject(user_id, conversation_id)
            if user_id and conversation_id
            else ""
        )
        await self._client.eval(
            SET_CONTEXT_VIEW_SCRIPT,
            7,
            self._context_view_key(key),
            self._context_view_seq_key(key),
            self._context_view_owner_key(key),
            self._context_view_user_index_key(user_id),
            self._context_view_conversation_owner_key(key),
            self._context_view_conversation_index_key(conversation_subject or "none"),
            self._context_view_lifecycle_owner_key(key),
            serialized,
            ttl_seconds,
            user_id,
            key,
            CONTEXT_VIEW_USER_INDEX_PREFIX,
            conversation_subject,
            CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX,
            LIFECYCLE_CONTEXT_INDEX_PREFIX,
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
            5,
            self._context_view_key(key),
            self._context_view_seq_key(key),
            self._context_view_owner_key(key),
            self._context_view_user_index_key(user_id),
            self._context_view_lifecycle_owner_key(key),
            json_utils.dumps(context_view, sort_keys=True),
            ttl_seconds,
            monotonic_seq,
            user_id,
            key,
            CONTEXT_VIEW_USER_INDEX_PREFIX,
            LIFECYCLE_CONTEXT_INDEX_PREFIX,
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

    async def set_context_view_if_newer_for_lifecycle(
        self,
        key: str,
        context_view: dict[str, Any],
        ttl_seconds: int,
        monotonic_seq: int,
        *,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> bool:
        user_id = extract_context_view_user_id(context_view)
        if user_id is None:
            return False
        conversation_id = extract_context_view_conversation_id(context_view)
        conversation_subject = (
            self._conversation_subject(user_id, conversation_id)
            if conversation_id is not None
            else ""
        )
        conversation_index_key = (
            self._context_view_conversation_index_key(conversation_subject)
            if conversation_subject
            else self._context_view_conversation_index_key("none")
        )
        result = await self._client.eval(
            SET_CONTEXT_VIEW_IF_NEWER_FOR_LIFECYCLE_SCRIPT,
            9,
            self._context_view_key(key),
            self._context_view_seq_key(key),
            self._context_view_owner_key(key),
            self._context_view_user_index_key(user_id),
            self._context_view_conversation_owner_key(key),
            conversation_index_key,
            self._lifecycle_mirror_key(lifecycle_cleanup_key),
            self._lifecycle_context_index_key(lifecycle_cleanup_key),
            self._context_view_lifecycle_owner_key(key),
            json_utils.dumps(context_view, sort_keys=True),
            ttl_seconds,
            monotonic_seq,
            user_id,
            key,
            CONTEXT_VIEW_USER_INDEX_PREFIX,
            f"active:{lifecycle_epoch}",
            conversation_subject,
            CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX,
            lifecycle_cleanup_key,
            LIFECYCLE_CONTEXT_INDEX_PREFIX,
        )
        return int(result or 0) == 1

    async def delete_context_view(self, key: str) -> None:
        await self._client.eval(
            DELETE_CONTEXT_VIEW_SCRIPT,
            5,
            self._context_view_key(key),
            self._context_view_seq_key(key),
            self._context_view_owner_key(key),
            self._context_view_conversation_owner_key(key),
            self._context_view_lifecycle_owner_key(key),
            key,
            CONTEXT_VIEW_USER_INDEX_PREFIX,
            CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX,
            LIFECYCLE_CONTEXT_INDEX_PREFIX,
            "",
            "",
        )

    async def delete_context_views_for_user(self, user_id: str) -> int:
        cache_keys = await self._client.smembers(
            self._context_view_user_index_key(user_id)
        )
        deleted = 0
        for cache_key in cache_keys:
            normalized_key = str(cache_key)
            deleted += int(
                await self._client.eval(
                    DELETE_CONTEXT_VIEW_SCRIPT,
                    5,
                    self._context_view_key(normalized_key),
                    self._context_view_seq_key(normalized_key),
                    self._context_view_owner_key(normalized_key),
                    self._context_view_conversation_owner_key(normalized_key),
                    self._context_view_lifecycle_owner_key(normalized_key),
                    normalized_key,
                    CONTEXT_VIEW_USER_INDEX_PREFIX,
                    CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX,
                    LIFECYCLE_CONTEXT_INDEX_PREFIX,
                    user_id,
                    "",
                )
                or 0
            )
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
            normalized_key = str(cache_key)
            deleted += int(
                await self._client.eval(
                    DELETE_CONTEXT_VIEW_SCRIPT,
                    5,
                    self._context_view_key(normalized_key),
                    self._context_view_seq_key(normalized_key),
                    self._context_view_owner_key(normalized_key),
                    self._context_view_conversation_owner_key(normalized_key),
                    self._context_view_lifecycle_owner_key(normalized_key),
                    normalized_key,
                    CONTEXT_VIEW_USER_INDEX_PREFIX,
                    CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX,
                    LIFECYCLE_CONTEXT_INDEX_PREFIX,
                    user_id,
                    conversation_subject,
                )
                or 0
            )
        return deleted

    async def purge_user_jobs(self, user_id: str) -> int:
        return await self._purge_jobs(
            lambda payload: _job_matches_user(payload, user_id),
            list_user_id=user_id,
        )

    async def purge_conversation_jobs(self, user_id: str, conversation_id: str) -> int:
        return await self._purge_jobs(
            lambda payload: (
                _job_matches_user(payload, user_id)
                and _job_matches_conversation(payload, conversation_id)
            ),
            list_user_id=user_id,
            list_conversation_id=conversation_id,
        )

    async def enqueue_job(self, queue_name: str, payload: dict[str, Any]) -> None:
        await self._client.rpush(
            self._queue_key(queue_name),
            json_utils.dumps(payload, sort_keys=True),
        )

    async def publish_lifecycle_diagnostic(
        self,
        queue_name: str,
        payload: dict[str, Any],
        *,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> str | None:
        delivery_id = f"dlq_{uuid4().hex}"
        serialized = json_utils.dumps(
            _wrap_lifecycle_diagnostic(
                payload,
                lifecycle_cleanup_key=lifecycle_cleanup_key,
                lifecycle_epoch=lifecycle_epoch,
                delivery_id=delivery_id,
            ),
            sort_keys=True,
        )
        published = await self._client.eval(
            PUBLISH_LIFECYCLE_DIAGNOSTIC_SCRIPT,
            3,
            self._lifecycle_mirror_key(lifecycle_cleanup_key),
            self._lifecycle_diagnostic_index_key(lifecycle_cleanup_key),
            self._queue_key(queue_name),
            f"active:{lifecycle_epoch}",
            serialized,
        )
        return delivery_id if int(published or 0) == 1 else None

    async def dequeue_job(
        self,
        queue_name: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raw_payload = await self._client.lpop(self._queue_key(queue_name))
            if raw_payload is None:
                return None
            return await self._decode_dequeued_job(queue_name, str(raw_payload))
        # None means "block forever", matching the in-process backend semantics.
        timeout = 0 if timeout_seconds is None else max(1, ceil(timeout_seconds))
        item = await self._client.blpop(
            self._queue_key(queue_name),
            timeout=timeout,
        )
        if item is None:
            return None
        _, raw_payload = item
        return await self._decode_dequeued_job(queue_name, str(raw_payload))

    async def stream_add(self, stream_name: str, payload: dict[str, Any]) -> str:
        message_id = await self._client.xadd(
            stream_name,
            {"payload": json_utils.dumps(payload, sort_keys=True)},
        )
        self._stream_add_counts[stream_name] = (
            self._stream_add_counts.get(stream_name, 0) + 1
        )
        return message_id

    async def prepare_lifecycle_mirror(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        nonce: str,
    ) -> str:
        preparing = f"preparing:{lifecycle_epoch}:{nonce}"
        result = await self._client.eval(
            PREPARE_LIFECYCLE_MIRROR_SCRIPT,
            1,
            self._lifecycle_mirror_key(lifecycle_cleanup_key),
            preparing,
            f"preparing:{lifecycle_epoch}:",
        )
        return str(result)

    async def activate_lifecycle_mirror(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        nonce: str,
    ) -> bool:
        result = await self._client.eval(
            ACTIVATE_LIFECYCLE_MIRROR_SCRIPT,
            1,
            self._lifecycle_mirror_key(lifecycle_cleanup_key),
            f"preparing:{lifecycle_epoch}:{nonce}",
            f"active:{lifecycle_epoch}",
        )
        return int(result or 0) == 1

    async def publish_job_notification(
        self,
        stream_name: str,
        payload: dict[str, Any],
        *,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> str | None:
        message_id = await self._client.eval(
            PUBLISH_JOB_NOTIFICATION_SCRIPT,
            4,
            self._lifecycle_mirror_key(lifecycle_cleanup_key),
            self._lifecycle_delivery_index_key(lifecycle_cleanup_key),
            stream_name,
            self._lifecycle_delivery_owner_key(stream_name),
            f"active:{lifecycle_epoch}",
            json_utils.dumps(payload, sort_keys=True),
            lifecycle_cleanup_key,
        )
        if not message_id:
            return None
        self._stream_add_counts[stream_name] = (
            self._stream_add_counts.get(stream_name, 0) + 1
        )
        return str(message_id)

    async def revoke_lifecycle_and_purge_notifications(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        *,
        group_name: str,
    ) -> int:
        result = await self._client.eval(
            REVOKE_LIFECYCLE_AND_PURGE_SCRIPT,
            6,
            self._lifecycle_mirror_key(lifecycle_cleanup_key),
            self._lifecycle_delivery_index_key(lifecycle_cleanup_key),
            self._lifecycle_recent_index_key(lifecycle_cleanup_key),
            self._lifecycle_context_index_key(lifecycle_cleanup_key),
            self._lifecycle_diagnostic_index_key(lifecycle_cleanup_key),
            self._lifecycle_lock_index_key(
                lifecycle_cleanup_key,
                lifecycle_epoch,
            ),
            f"revoked:{lifecycle_epoch}",
            group_name,
            LIFECYCLE_DELIVERY_OWNER_PREFIX,
            "recent_window:",
            CONTEXT_VIEW_PREFIX,
            CONTEXT_VIEW_SEQ_PREFIX,
            CONTEXT_VIEW_OWNER_PREFIX,
            CONTEXT_VIEW_CONVERSATION_OWNER_PREFIX,
            CONTEXT_VIEW_USER_INDEX_PREFIX,
            CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX,
            RECENT_WINDOW_CACHE_IDENTITY_PREFIX,
            lifecycle_epoch,
            lifecycle_cleanup_key,
            RECENT_WINDOW_USER_INDEX_PREFIX,
            CONTEXT_VIEW_LIFECYCLE_OWNER_PREFIX,
        )
        return int(result or 0)

    async def stream_read(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        *,
        count: int,
        block_ms: int | None,
    ) -> list[StreamMessage]:
        redis_block_ms = (
            0 if block_ms is None else (None if block_ms <= 0 else int(block_ms))
        )
        try:
            response = await self._client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={stream_name: ">"},
                count=count,
                block=redis_block_ms,
            )
        except ResponseError as exc:
            if not self._is_missing_consumer_group(exc):
                raise
            await self.stream_ensure_group(stream_name, group_name)
            response = await self._client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={stream_name: ">"},
                count=count,
                block=redis_block_ms,
            )
        messages: list[StreamMessage] = []
        for _stream, entries in response:
            for message_id, fields in entries:
                normalized_fields = self._normalize_stream_fields(fields)
                raw_payload = normalized_fields.get("payload", "{}")
                messages.append(
                    StreamMessage(
                        message_id=message_id,
                        payload=json_utils.loads(raw_payload),
                        delivery_count=1,
                    )
                )
                self._stream_read_counts[stream_name] = (
                    self._stream_read_counts.get(stream_name, 0) + 1
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
        try:
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
        except ResponseError as exc:
            if not self._is_missing_consumer_group(exc):
                raise
            await self.stream_ensure_group(stream_name, group_name)
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
        delivery_counts = await self._pending_delivery_counts(
            stream_name, group_name, entries
        )
        for message_id, fields in entries:
            normalized_fields = self._normalize_stream_fields(fields)
            raw_payload = normalized_fields.get("payload", "{}")
            messages.append(
                StreamMessage(
                    message_id=message_id,
                    payload=json_utils.loads(raw_payload),
                    delivery_count=delivery_counts.get(message_id, 2),
                )
            )
            self._stream_claim_counts[stream_name] = (
                self._stream_claim_counts.get(stream_name, 0) + 1
            )
        return messages

    async def stream_ack(
        self, stream_name: str, group_name: str, message_id: str
    ) -> None:
        removed = await self._client.eval(
            ACK_INDEXED_STREAM_MESSAGE_SCRIPT,
            2,
            stream_name,
            self._lifecycle_delivery_owner_key(stream_name),
            group_name,
            message_id,
            LIFECYCLE_DELIVERY_INDEX_PREFIX,
        )
        if int(removed or 0) > 0:
            self._stream_ack_counts[stream_name] = (
                self._stream_ack_counts.get(stream_name, 0) + 1
            )

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

    async def drain_snapshot(self) -> StorageDrainSnapshot:
        queued_by_stream: dict[str, int] = {}
        pending_by_stream: dict[str, int] = {}
        for stream_name, group_name in self._stream_groups:
            pending_count, lag_count = await self._group_backlog(
                stream_name, group_name
            )
            queued_by_stream[stream_name] = max(
                queued_by_stream.get(stream_name, 0),
                lag_count,
            )
            pending_by_stream[stream_name] = (
                pending_by_stream.get(stream_name, 0) + pending_count
            )
        return StorageDrainSnapshot(
            queued_by_stream=queued_by_stream,
            pending_by_stream=pending_by_stream,
            added_by_stream=dict(self._stream_add_counts),
            read_by_stream=dict(self._stream_read_counts),
            claimed_by_stream=dict(self._stream_claim_counts),
            acked_by_stream=dict(self._stream_ack_counts),
        )

    async def drain(
        self,
        timeout_seconds: float = 30.0,
        *,
        idle_timeout_seconds: float | None = None,
        progress_interval_seconds: float = 0.0,
        progress_callback: DrainProgressCallback | None = None,
    ) -> bool:
        if not self._stream_groups:
            return True

        timeout = max(0.0, timeout_seconds)
        idle_timeout = (
            None if idle_timeout_seconds is None else max(0.0, idle_timeout_seconds)
        )
        started_at = monotonic()
        deadline = started_at + timeout
        last_progress_at = started_at
        last_marker: tuple[tuple[tuple[str, int], ...], ...] | None = None
        progress_interval = max(0.0, progress_interval_seconds)
        next_progress_at = started_at + progress_interval
        stable_started_at: float | None = None
        while True:
            now = monotonic()
            snapshot = (await self.drain_snapshot()).with_timing(
                elapsed_seconds=now - started_at,
                idle_seconds=now - last_progress_at,
                timeout_seconds=timeout,
                idle_timeout_seconds=idle_timeout,
            )
            marker = snapshot.progress_marker()
            if last_marker is None:
                last_marker = marker
            elif marker != last_marker:
                last_marker = marker
                last_progress_at = now
                snapshot = snapshot.with_timing(
                    elapsed_seconds=now - started_at,
                    idle_seconds=0.0,
                    timeout_seconds=timeout,
                    idle_timeout_seconds=idle_timeout,
                )

            if snapshot.drained:
                if stable_started_at is None:
                    stable_started_at = monotonic()
                elif monotonic() - stable_started_at >= DRAIN_STABLE_WINDOW_SECONDS:
                    return True
            else:
                stable_started_at = None

            if progress_callback is not None and now >= next_progress_at:
                if await emit_drain_progress(progress_callback, snapshot):
                    last_progress_at = now
                next_progress_at = now + max(progress_interval, 0.01)
            if (
                idle_timeout is not None
                and monotonic() - last_progress_at >= idle_timeout
            ):
                return False
            if monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)

    async def remember_dedupe(
        self,
        key: str,
        ttl_seconds: int,
    ) -> bool:
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

    async def acquire_lock(
        self,
        key: str,
        ttl_seconds: int,
        *,
        lifecycle_cleanup_key: str | None = None,
        lifecycle_epoch: str | None = None,
        job_id: str | None = None,
        execution_fence: int | None = None,
    ) -> str | None:
        lifecycle = _validated_lifecycle_coordinates(
            lifecycle_cleanup_key,
            lifecycle_epoch,
        )
        job_scope = _validated_job_lock_scope(
            job_id,
            execution_fence,
            lifecycle=lifecycle,
        )
        token = uuid4().hex
        if lifecycle is not None:
            cleanup_key, epoch = lifecycle
            result = await self._client.eval(
                ACQUIRE_LIFECYCLE_LOCK_SCRIPT,
                4,
                self._lifecycle_mirror_key(cleanup_key),
                self._lifecycle_lock_key(key, cleanup_key, epoch),
                self._lifecycle_lock_index_key(cleanup_key, epoch),
                self._lifecycle_lock_high_water_key(
                    key,
                    cleanup_key,
                    epoch,
                    job_scope[0] if job_scope is not None else "",
                ),
                f"active:{epoch}",
                token,
                ttl_seconds,
                job_scope[0] if job_scope is not None else "",
                job_scope[1] if job_scope is not None else "",
            )
            return token if int(result or 0) == 1 else None
        result = await self._client.set(
            f"lock:{key}",
            token,
            ex=ttl_seconds,
            nx=True,
        )
        return token if result else None

    async def release_lock(
        self,
        key: str,
        token: str,
        *,
        lifecycle_cleanup_key: str | None = None,
        lifecycle_epoch: str | None = None,
        job_id: str | None = None,
        execution_fence: int | None = None,
    ) -> None:
        lifecycle = _validated_lifecycle_coordinates(
            lifecycle_cleanup_key,
            lifecycle_epoch,
        )
        job_scope = _validated_job_lock_scope(
            job_id,
            execution_fence,
            lifecycle=lifecycle,
        )
        if lifecycle is not None:
            cleanup_key, epoch = lifecycle
            await self._client.eval(
                RELEASE_LIFECYCLE_LOCK_SCRIPT,
                2,
                self._lifecycle_lock_key(key, cleanup_key, epoch),
                self._lifecycle_lock_index_key(cleanup_key, epoch),
                token,
                job_scope[0] if job_scope is not None else "",
                job_scope[1] if job_scope is not None else "",
            )
            return
        await self._client.eval(RELEASE_LOCK_SCRIPT, 1, f"lock:{key}", token)

    async def purge_legacy_transient_state(
        self,
        database_path: str,
        user_id: str,
    ) -> LegacyTransientPurgeResult:
        del database_path
        cache_generation_deleted = 0
        malformed_candidates = 0
        async for redis_key in self._client.scan_iter(match="cachegen:*"):
            normalized_key = str(redis_key)
            if normalized_key == f"cachegen:{user_id}":
                cache_generation_deleted += int(
                    await self._client.delete(normalized_key) or 0
                )
                continue
            parts = normalized_key.split(":", 2)
            if len(parts) == 2 and parts[1]:
                malformed_candidates += 1
                continue
            if len(parts) != 3:
                malformed_candidates += 1
                continue
            namespace = parts[1]
            candidate_user_id = parts[2]
            if candidate_user_id == user_id:
                cache_generation_deleted += int(
                    await self._client.delete(normalized_key) or 0
                )
            elif not _is_twelve_hex_namespace(namespace) or not candidate_user_id:
                malformed_candidates += 1

        async for redis_key in self._client.scan_iter(
            match=f"{RECENT_WINDOW_CACHE_IDENTITY_PREFIX}*"
        ):
            normalized_key = str(redis_key)
            if await self._client.type(normalized_key) != "string":
                malformed_candidates += 1
                continue
            logical_key = normalized_key.removeprefix(
                RECENT_WINDOW_CACHE_IDENTITY_PREFIX
            )
            for _attempt in range(3):
                identity_raw = await self._client.get(normalized_key)
                if identity_raw is None:
                    break
                identity = self._decode_current_recent_identity(
                    logical_key,
                    identity_raw,
                )
                if identity is None:
                    malformed_candidates += 1
                    break
                if identity.user_id != user_id:
                    break
                result = int(
                    await self._client.eval(
                        PURGE_ORPHAN_RECENT_IDENTITY_SCRIPT,
                        4,
                        normalized_key,
                        f"recent_window:{logical_key}",
                        self._lifecycle_recent_index_key(
                            identity.lifecycle_cleanup_key
                        ),
                        self._recent_window_user_index_key(identity.user_id),
                        identity_raw,
                        f"r\x1f{logical_key}",
                        logical_key,
                    )
                )
                if result >= 0:
                    break
            else:
                malformed_candidates += 1

        recent_windows_deleted = 0
        async for redis_key in self._client.scan_iter(match="recent_window:*"):
            normalized_key = str(redis_key)
            if await self._client.type(normalized_key) != "string":
                malformed_candidates += 1
                continue
            logical_key = normalized_key.removeprefix("recent_window:")
            identity_key = self._recent_window_cache_identity_key(logical_key)
            for _attempt in range(3):
                identity_type = await self._client.type(identity_key)
                if identity_type not in {"none", "string"}:
                    malformed_candidates += 1
                    break
                identity_raw = await self._client.get(identity_key)
                identity = self._decode_current_recent_identity(
                    logical_key,
                    identity_raw,
                )
                lifecycle_cleanup_key = (
                    identity.lifecycle_cleanup_key if identity is not None else ""
                )
                result = int(
                    await self._client.eval(
                        PURGE_LEGACY_RECENT_WINDOW_SCRIPT,
                        5,
                        normalized_key,
                        identity_key,
                        self._lifecycle_mirror_key(lifecycle_cleanup_key),
                        self._lifecycle_recent_index_key(lifecycle_cleanup_key),
                        self._recent_window_user_index_key(
                            identity.user_id if identity is not None else ""
                        ),
                        "1" if identity_raw is not None else "0",
                        identity_raw or "",
                        "1" if identity is not None else "0",
                        (
                            f"active:{identity.lifecycle_epoch}"
                            if identity is not None
                            else ""
                        ),
                        f"r\x1f{logical_key}",
                        logical_key,
                    )
                )
                if result >= 0:
                    recent_windows_deleted += result
                    break
            else:
                malformed_candidates += 1

        recent_user_index = self._recent_window_user_index_key(user_id)
        for indexed_logical_key in await self._client.smembers(recent_user_index):
            logical_key = str(indexed_logical_key)
            identity_key = self._recent_window_cache_identity_key(logical_key)
            for _attempt in range(3):
                identity_type = await self._client.type(identity_key)
                if identity_type not in {"none", "string"}:
                    malformed_candidates += 1
                    break
                identity_raw = await self._client.get(identity_key)
                identity = self._decode_current_recent_identity(
                    logical_key,
                    identity_raw,
                )
                lifecycle_cleanup_key = (
                    identity.lifecycle_cleanup_key if identity is not None else ""
                )
                result = int(
                    await self._client.eval(
                        PRUNE_RECENT_USER_MEMBER_SCRIPT,
                        5,
                        recent_user_index,
                        f"recent_window:{logical_key}",
                        identity_key,
                        self._lifecycle_mirror_key(lifecycle_cleanup_key),
                        self._lifecycle_recent_index_key(lifecycle_cleanup_key),
                        "1" if identity_raw is not None else "0",
                        identity_raw or "",
                        "1" if identity is not None else "0",
                        (
                            "1"
                            if identity is not None and identity.user_id == user_id
                            else "0"
                        ),
                        (
                            f"active:{identity.lifecycle_epoch}"
                            if identity is not None
                            else ""
                        ),
                        f"r\x1f{logical_key}",
                        logical_key,
                    )
                )
                if result >= 0:
                    break
            else:
                malformed_candidates += 1

        async for redis_key in self._client.scan_iter(match="context_view_owner:*"):
            normalized_key = str(redis_key)
            if await self._client.type(normalized_key) != "string":
                malformed_candidates += 1
                continue
            logical_key = normalized_key.removeprefix(CONTEXT_VIEW_OWNER_PREFIX)
            for _attempt in range(3):
                owner = await self._client.get(normalized_key)
                if owner is None:
                    break
                if not owner:
                    malformed_candidates += 1
                    break
                if owner != user_id:
                    break
                result = int(
                    await self._client.eval(
                        PURGE_ORPHAN_CONTEXT_OWNER_SCRIPT,
                        5,
                        normalized_key,
                        self._context_view_key(logical_key),
                        self._context_view_seq_key(logical_key),
                        self._context_view_conversation_owner_key(logical_key),
                        self._context_view_lifecycle_owner_key(logical_key),
                        owner,
                        logical_key,
                        user_id,
                        CONTEXT_VIEW_USER_INDEX_PREFIX,
                        CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX,
                        LIFECYCLE_CONTEXT_INDEX_PREFIX,
                    )
                )
                if result >= 0:
                    break
            else:
                malformed_candidates += 1

        context_views_deleted = 0
        async for redis_key in self._client.scan_iter(match="context_view:*"):
            normalized_key = str(redis_key)
            if await self._client.type(normalized_key) != "string":
                malformed_candidates += 1
                continue
            logical_key = normalized_key.removeprefix(CONTEXT_VIEW_PREFIX)
            lifecycle_owner_key = self._context_view_lifecycle_owner_key(logical_key)
            for _attempt in range(3):
                raw = await self._client.get(normalized_key)
                if raw is None:
                    break
                lifecycle_owner = await self._client.get(lifecycle_owner_key)
                action = "preserve"
                try:
                    decoded = json_utils.loads(raw)
                except ValueError:
                    decoded = None
                if not isinstance(decoded, dict):
                    action = "malformed"
                else:
                    owner_user_id = extract_context_view_user_id(decoded)
                    if owner_user_id is None:
                        action = "malformed"
                    elif owner_user_id == user_id:
                        action = "delete"
                result = int(
                    await self._client.eval(
                        PURGE_LEGACY_CONTEXT_VIEW_SCRIPT,
                        7,
                        normalized_key,
                        self._context_view_seq_key(logical_key),
                        self._context_view_owner_key(logical_key),
                        self._context_view_conversation_owner_key(logical_key),
                        lifecycle_owner_key,
                        self._lifecycle_mirror_key(lifecycle_owner or ""),
                        self._lifecycle_context_index_key(lifecycle_owner or ""),
                        raw,
                        "1" if lifecycle_owner is not None else "0",
                        lifecycle_owner or "",
                        action,
                        logical_key,
                        CONTEXT_VIEW_USER_INDEX_PREFIX,
                        CONTEXT_VIEW_CONVERSATION_INDEX_PREFIX,
                        user_id,
                    )
                )
                if result < 0:
                    continue
                if result == 1:
                    context_views_deleted += 1
                elif result == 2:
                    malformed_candidates += 1
                break
            else:
                malformed_candidates += 1
        await self._client.eval(
            PRUNE_CONTEXT_USER_INDEX_SCRIPT,
            1,
            self._context_view_user_index_key(user_id),
            CONTEXT_VIEW_PREFIX,
            CONTEXT_VIEW_OWNER_PREFIX,
            user_id,
        )

        stream_entries_deleted = 0
        queue_entries_deleted = 0
        deferred_entries_deleted = 0
        for normalized_key in sorted(_ATAGIA_JOB_STREAM_NAMES):
            key_type = await self._client.type(normalized_key)
            if key_type == "none":
                continue
            if key_type != "stream":
                malformed_candidates += 1
                continue
            result = await self._client.eval(
                PURGE_LEGACY_STREAM_ENTRIES_SCRIPT,
                1,
                normalized_key,
                user_id,
                LIFECYCLE_MIRROR_PREFIX,
                LIFECYCLE_DELIVERY_INDEX_PREFIX,
                LIFECYCLE_DELIVERY_OWNER_PREFIX,
            )
            stream_entries_deleted += int(result[0])
            malformed_candidates += int(result[1])

        for queue_name in sorted(_LEGACY_ATAGIA_LIST_QUEUE_NAMES):
            normalized_key = f"queue:{queue_name}"
            key_type = await self._client.type(normalized_key)
            if key_type == "none":
                continue
            if key_type != "list":
                malformed_candidates += 1
                continue
            result = await self._client.eval(
                PURGE_LEGACY_DEAD_LETTER_ITEMS_SCRIPT,
                1,
                normalized_key,
                user_id,
                LIFECYCLE_MIRROR_PREFIX,
                LIFECYCLE_DIAGNOSTIC_INDEX_PREFIX,
            )
            queue_entries_deleted += int(result[0])
            malformed_candidates += int(result[1])

        for stream_name in sorted(_ATAGIA_JOB_STREAM_NAMES):
            normalized_key = f"stream_deferred:{stream_name}"
            key_type = await self._client.type(normalized_key)
            if key_type == "none":
                continue
            if key_type != "zset":
                malformed_candidates += 1
                continue
            result = await self._client.eval(
                PURGE_LEGACY_DEFERRED_ITEMS_SCRIPT,
                1,
                normalized_key,
                user_id,
            )
            deferred_entries_deleted += int(result[0])
            malformed_candidates += int(result[1])

        legacy_dedupe_deleted = 0
        async for redis_key in self._client.scan_iter(match="dedupe:*"):
            normalized_key = str(redis_key)
            logical_key = normalized_key.removeprefix("dedupe:")
            if _is_legacy_extractor_dedupe_key_for_user(logical_key, user_id):
                legacy_dedupe_deleted += int(
                    await self._client.delete(normalized_key) or 0
                )

        # No historical generic lock shape injectively identifies a user.
        # Opaque, current, and other-user locks are outside this candidate set.
        legacy_locks_deleted = 0

        return LegacyTransientPurgeResult(
            cache_generation_deleted=cache_generation_deleted,
            recent_windows_deleted=recent_windows_deleted,
            context_views_deleted=context_views_deleted,
            stream_entries_deleted=stream_entries_deleted,
            queue_entries_deleted=queue_entries_deleted,
            deferred_entries_deleted=deferred_entries_deleted,
            legacy_dedupe_deleted=legacy_dedupe_deleted,
            legacy_locks_deleted=legacy_locks_deleted,
            malformed_candidates=malformed_candidates,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _purge_jobs(
        self,
        should_drop: Any,
        *,
        list_user_id: str,
        list_conversation_id: str | None = None,
    ) -> int:
        purged = 0
        async for key in self._client.scan_iter(match=f"{ATAGIA_QUEUE_PREFIX}*"):
            queue_key = str(key)
            purged += int(
                await self._client.eval(
                    PURGE_LIST_JOBS_SCRIPT,
                    1,
                    queue_key,
                    list_user_id,
                    list_conversation_id or "",
                )
                or 0
            )

        visited_streams: set[str] = set()
        for stream_name, group_name in list(self._stream_groups):
            visited_streams.add(stream_name)
            purged += await self._purge_stream_entries(
                stream_name, group_name, should_drop
            )
        for stream_name in sorted(_ATAGIA_JOB_STREAM_NAMES - visited_streams):
            try:
                key_type = await self._client.type(stream_name)
            except Exception:
                continue
            if key_type != "stream":
                continue
            purged += await self._purge_stream_entries(stream_name, None, should_drop)
        return purged

    @staticmethod
    def _queue_key(queue_name: str) -> str:
        return f"{ATAGIA_QUEUE_PREFIX}{queue_name}"

    @staticmethod
    def _lifecycle_mirror_key(lifecycle_cleanup_key: str) -> str:
        return f"{LIFECYCLE_MIRROR_PREFIX}{lifecycle_cleanup_key}"

    @staticmethod
    def _lifecycle_delivery_index_key(lifecycle_cleanup_key: str) -> str:
        return f"{LIFECYCLE_DELIVERY_INDEX_PREFIX}{lifecycle_cleanup_key}"

    @staticmethod
    def _lifecycle_diagnostic_index_key(lifecycle_cleanup_key: str) -> str:
        return f"{LIFECYCLE_DIAGNOSTIC_INDEX_PREFIX}{lifecycle_cleanup_key}"

    @staticmethod
    def _lifecycle_transient_namespace(
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> str:
        return canonical_json_hash(
            [lifecycle_cleanup_key, lifecycle_epoch],
        )

    @classmethod
    def _lifecycle_lock_key(
        cls,
        key: str,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> str:
        return f"{LIFECYCLE_LOCK_PREFIX}" + canonical_json_hash(
            [
                cls._lifecycle_transient_namespace(
                    lifecycle_cleanup_key,
                    lifecycle_epoch,
                ),
                key,
            ]
        )

    @classmethod
    def _lifecycle_lock_index_key(
        cls,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
    ) -> str:
        return (
            f"{LIFECYCLE_LOCK_INDEX_PREFIX}"
            f"{cls._lifecycle_transient_namespace(lifecycle_cleanup_key, lifecycle_epoch)}"
        )

    @classmethod
    def _lifecycle_lock_high_water_key(
        cls,
        key: str,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        job_id: str,
    ) -> str:
        return f"{LIFECYCLE_LOCK_HIGH_WATER_PREFIX}" + canonical_json_hash(
            [
                cls._lifecycle_transient_namespace(
                    lifecycle_cleanup_key,
                    lifecycle_epoch,
                ),
                key,
                job_id,
            ]
        )

    @staticmethod
    def _lifecycle_recent_index_key(lifecycle_cleanup_key: str) -> str:
        return f"{LIFECYCLE_RECENT_INDEX_PREFIX}{lifecycle_cleanup_key}"

    @staticmethod
    def _recent_window_cache_identity_key(key: str) -> str:
        return f"{RECENT_WINDOW_CACHE_IDENTITY_PREFIX}{key}"

    @staticmethod
    def _recent_window_user_index_key(user_id: str) -> str:
        return f"{RECENT_WINDOW_USER_INDEX_PREFIX}{user_id}"

    @staticmethod
    def _recent_window_cache_identity(
        identity: RecentWindowIdentity,
    ) -> str:
        return json_utils.dumps(
            {
                "cache_revision": identity.cache_revision,
                "conversation_id": identity.conversation_id,
                "conversation_lifecycle_epoch": (identity.conversation_lifecycle_epoch),
                "conversation_source_revision": (identity.conversation_source_revision),
                "derivation_revision": identity.derivation_revision,
                "lifecycle_cleanup_key": identity.lifecycle_cleanup_key,
                "lifecycle_epoch": identity.lifecycle_epoch,
                "user_id": identity.user_id,
            },
            sort_keys=True,
        )

    @staticmethod
    def _decode_current_recent_identity(
        logical_key: str,
        identity_raw: str | None,
    ) -> RecentWindowIdentity | None:
        if identity_raw is None:
            return None
        try:
            decoded = json_utils.loads(identity_raw)
            if not isinstance(decoded, dict):
                return None
            identity = RecentWindowIdentity(**decoded)
        except (TypeError, ValueError):
            return None
        if not _recent_window_identity_is_valid(identity):
            return None
        if (
            build_recent_window_key(
                identity.user_id,
                identity.conversation_id,
            )
            != logical_key
        ):
            return None
        return identity

    @staticmethod
    def _lifecycle_context_index_key(lifecycle_cleanup_key: str) -> str:
        return f"{LIFECYCLE_CONTEXT_INDEX_PREFIX}{lifecycle_cleanup_key}"

    @staticmethod
    def _lifecycle_delivery_owner_key(stream_name: str) -> str:
        return f"{LIFECYCLE_DELIVERY_OWNER_PREFIX}{stream_name}"

    async def _decode_dequeued_job(
        self,
        queue_name: str,
        raw_payload: str,
    ) -> dict[str, Any]:
        decoded = json_utils.loads(raw_payload)
        metadata = _lifecycle_diagnostic_metadata(decoded)
        if metadata is not None:
            await self._client.srem(
                self._lifecycle_diagnostic_index_key(metadata["lifecycle_cleanup_key"]),
                f"{self._queue_key(queue_name)}\x1f{raw_payload}",
            )
        payload = _unwrap_lifecycle_diagnostic(decoded)
        if not isinstance(payload, dict):
            raise ValueError("Queued job payload must be a JSON object")
        return payload

    async def _purge_stream_entries(
        self,
        stream_name: str,
        group_name: str | None,
        should_drop: Any,
    ) -> int:
        purged = 0
        group_names = (
            [group_name]
            if group_name is not None
            else await self._stream_group_names(stream_name)
        )
        for message_id, fields in await self._client.xrange(stream_name, "-", "+"):
            normalized_fields = self._normalize_stream_fields(fields)
            try:
                payload = json_utils.loads(normalized_fields.get("payload", "{}"))
            except Exception:
                continue
            if should_drop(payload):
                for ack_group_name in group_names:
                    await self._client.xack(stream_name, ack_group_name, message_id)
                await self._client.xdel(stream_name, message_id)
                purged += 1
        return purged

    async def _stream_group_names(self, stream_name: str) -> list[str]:
        try:
            groups = await self._client.xinfo_groups(stream_name)
        except ResponseError:
            return []
        names: list[str] = []
        for group in groups:
            if isinstance(group, dict) and group.get("name") is not None:
                names.append(str(group["name"]))
        return names

    async def _sync_context_view_owner(
        self,
        *,
        key: str,
        user_id: str | None,
        previous_user_id: str | None,
        ttl_seconds: int,
    ) -> None:
        if previous_user_id and previous_user_id != user_id:
            await self._client.srem(
                self._context_view_user_index_key(previous_user_id), key
            )
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
        if (
            previous_conversation_subject
            and previous_conversation_subject != conversation_subject
        ):
            await self._client.srem(
                self._context_view_conversation_index_key(
                    previous_conversation_subject
                ),
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
    def _context_view_lifecycle_owner_key(key: str) -> str:
        return f"{CONTEXT_VIEW_LIFECYCLE_OWNER_PREFIX}{key}"

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

    async def _group_backlog(
        self, stream_name: str, group_name: str
    ) -> tuple[int, int]:
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

    @staticmethod
    def _is_missing_consumer_group(exc: BaseException) -> bool:
        return "NOGROUP" in str(exc).upper()
