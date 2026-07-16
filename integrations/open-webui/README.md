# Open WebUI Integration

Status: contract-tested against Open WebUI v0.9.0; live deployment smoke still
required.

`atagia_memory_filter.py` is a copyable Open WebUI Filter Function. Its host
contract is pinned to Open WebUI `v0.9.0`, commit
`f31768e20e5c6b4f6da0ef657877298b359936cf`. The contract fixture mirrors the
official `process_filter_functions` signature-based injection from
`backend/open_webui/utils/filter.py` at that commit (source SHA-256
`61e08fb796c042b295e0c6c570429f458c375bc2dec5e9a55e2901b6ed1c1e12`).

## Install

Import `atagia_memory_filter.py` from Admin Panel -> Functions, review it, save
it, then attach it globally or to selected models. Configure:

```text
enabled: true
base_url: http://127.0.0.1:8100
api_key: <ATAGIA_SERVICE_API_KEY>
installation_id: <stable unique ID for this Open WebUI deployment>
default_host_account_id: <stable account ID only for a single-user deployment>
default_user_id: <mapped Atagia user only for a single-user deployment>
default_conversation_id: <stable chat ID only for a single-chat deployment>
platform_id: open-webui
user_persona_id:
character_id:
mode: general_qa
memory_privacy_mode: balanced
fail_open: true
emit_debug_status: false
diagnostic_cache_max_entries: 512
diagnostic_cache_ttl_seconds: 900
```

`installation_id` is mandatory while enabled. The default identity fallbacks
are empty: Open WebUI must supply stable account/chat metadata, or the operator
must configure an explicit fallback only when that deployment truly has one
account/chat. In multi-user deployments, Open WebUI's trusted `__user__.id` is
both the host account identity and the default Atagia user mapping. Request or
body metadata cannot override it. A distinct `default_user_id` is accepted only
when `default_host_account_id` names that exact single trusted account; requests
from any other `__user__` fail before contacting Atagia. Do not configure one
shared fallback account for a multi-user deployment. The Atagia service key
remains a server-side Function valve and must not be exposed to browser code.

## Context Ownership And Correlation

The filter inserts one standalone system message bounded by the versioned
`ATAGIA:FILTER:MEMORY_CONTEXT` markers. On every inlet reentry it removes only
that owned message before inserting the current context. It never appends to or
edits unrelated system/user/assistant messages. Empty context, disablement, and
fail-open also remove stale owned context.

Open WebUI v0.9.0 passes the same supported `__metadata__` dictionary to filter
hooks. The filter stores only an opaque, high-entropy correlation token there.
Its scope remains in the filter's bounded in-process TTL/LRU store and outlet
accepts it only when the current trusted `__user__` matches the scope that
created it. Forged dictionaries, expired tokens, cross-user replay, and
conflicting metadata identity claims fail before a service-key request. No
context or message text is retained in correlation state.

Message identity is a canonical hash of integration kind, deployment, host
account, mapped Atagia user, host conversation/message, role, and generation.
Text is never an identity input. Native host IDs are preferred. Without one,
the chronological host-message ordinal is used in the explicit `live_event`
namespace. A generation that replaces an existing position omits synthetic
`source_seq` unless Open WebUI supplies a trustworthy one.

## Bounded Diagnostics And Deletion

`debug_state()` exposes status, IDs, optional ordinals, `has_context`, and a
generic error code only. It never retains a prompt preview or raw exception.
The in-process diagnostic cache is TTL-bounded and LRU-bounded by its valves.

When host lifecycle code deletes a chat or user, call:

```python
filter.delete_state(user_id="mapped-atagia-user", conversation_id="atagia-chat")
filter.delete_state(user_id="mapped-atagia-user")
```

Calling `delete_state()` without filters clears all diagnostic metadata. Open
WebUI v0.9.0 does not provide a Filter deletion hook, so deployment glue must
invoke this method from its chat/user deletion lifecycle; this repository does
not claim automatic deletion without that wiring. TTL remains the fallback.

## Outlet Caveat

Open WebUI does not run `outlet()` for every API/direct flow. When guaranteed
assistant-response persistence matters for API-direct use, use the Atagia
OpenAI-compatible proxy rather than relying on the filter.
