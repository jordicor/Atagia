# Host Application Sidecar Integration

Atagia's canonical runtime integration contract is the Python package module
`atagia.integrations`. The repository-level `integrations/` directory is only
for copyable host scaffolds and notes.

## Public Runtime API

Production host code should import reusable helpers from `atagia.integrations`:

- `SidecarBridge`
- `SidecarBridgeConfig`
- `SidecarBridgeError`
- `IngestOrigin`
- `ConfirmationStrategy`
- `MemoryPrivacyMode`
- `WorkerControlMode`
- `build_injection_decision`
- `context_messages_for_provider`
- `extract_context_system_prompt`
- `extract_context_message_id`
- `message_to_text`

Aurvek-specific naming conventions are available from the same package:

- `AURVEK_PLATFORM_ID`
- `aurvek_user_id(id)`
- `aurvek_conversation_id(id)`
- `aurvek_message_id(id)`
- `aurvek_prompt_character_id(prompt_id)`

## Host Loop

1. Map verified host IDs to stable Atagia IDs.
2. Ensure the Atagia user and conversation exist.
3. Fetch context before the host LLM call.
4. Inject `context.system_prompt` with `build_injection_decision`.
5. Drop duplicated long host history when Atagia is active.
6. Persist the assistant response after generation.
7. Keep host fallback behavior if Atagia is disabled or fails.

`context.initial_context_package` is a safe diagnostic summary for the prepared
context package read on that turn. Hosts may log it for rollout analysis, but
should not display it as user-facing content. Package hits add prepared
orientation to `context.system_prompt`; misses, stale rows, signature mismatches,
or disabled reads fall open to the normal live context path. A package hit does
not mean query-specific retrieval was skipped.

## Coordinate Fields

Hosts should pass stable IDs from their own data model instead of asking Atagia
to infer semantic boundaries from text:

- `platform_id`: software host/device boundary, such as `aurvek-web` or
  `sillytavern-desktop`.
- `character_id`: long-running prompt, character, project, campaign, or client
  identity for cross-chat continuity.
- `space_id`: project/folder/room/capsule boundary when the host has a
  first-class area that should focus or isolate memory.
- `active_presence_id`: explicit voice, character, facet, human, or source
  speaker when the host already has a durable Presence record.
- `mind_id` and `mind_topology`: perspectival memory owner and topology for
  unimind, multi-mind, or OjoCentauri deployments.
- `embodiment_id`: concrete body/device whose capabilities, sensors, location,
  or local constraints should not automatically transfer elsewhere.
- `realm_id`: world, fiction, simulation, game save, campaign, or domain of
  applicability.

Leave a coordinate unset when the host has no meaningful boundary for it.
Atagia still starts every retrieval with `user_id`, then applies the declared
coordinates and policy gates before ranking.

## Aurvek Example

```python
from atagia.integrations import (
    AURVEK_PLATFORM_ID,
    SidecarBridge,
    SidecarBridgeConfig,
    MemoryPrivacyMode,
    aurvek_conversation_id,
    aurvek_message_id,
    aurvek_prompt_character_id,
    aurvek_user_id,
)

bridge = SidecarBridge(
    SidecarBridgeConfig(
        enabled=True,
        platform_id=AURVEK_PLATFORM_ID,
        mode="personal_assistant",
    )
)

context = await bridge.get_context_for_turn(
    user_id=aurvek_user_id(7),
    conversation_id=aurvek_conversation_id(91),
    message_text="What did we decide?",
    platform_id=AURVEK_PLATFORM_ID,
    mode="personal_assistant",
    character_id=aurvek_prompt_character_id(3),
    message_id=aurvek_message_id(123),
    source_seq=123,
    memory_privacy_mode=MemoryPrivacyMode.BALANCED,
)
```

`message_id` is optional across `get_context`, `ingest_message`, and
`add_response`. If the same `message_id` is retried with the same role and text,
Atagia treats it as an idempotent replay and does not create a duplicate
message. If the same `message_id` is reused with different content, role, user,
or conversation, Atagia returns a conflict.

For historical backfills, pass `source_seq` as the host conversation-local
message order. Atagia uses it as the internal `messages.seq`, so a retry of an
old failed message lands back in its chronological gap instead of being appended
after newer synced messages. `source_seq` is intentionally the only public name
for this field; do not also introduce `host_seq` in host adapters.

Also pass `ingest_origin="backfill"` for historical sync/import paths:

```python
await bridge.ingest_message(
    user_id=aurvek_user_id(7),
    conversation_id=aurvek_conversation_id(91),
    role="user",
    text="Historical message",
    platform_id=AURVEK_PLATFORM_ID,
    character_id=aurvek_prompt_character_id(3),
    message_id=aurvek_message_id(123),
    source_seq=123,
    ingest_origin="backfill",
)
```

Default behavior is `ingest_origin="live_turn"` with
`confirmation_strategy="live_prompt_allowed"`. For `backfill` and
`admin_import`, Atagia defaults to `confirmation_strategy="admin_review_only"`.
Sensitive imported candidates are stored as `review_required`, not
`pending_user_confirmation`, so hosts do not end up with user prompts for old
history in the default `balanced` privacy mode.

`mode` remains the retrieval profile, and `operational_profile` remains an
operational runtime profile. Neither should be used as a privacy/trust selector
by host apps.

Use `memory_privacy_mode` for the user's storage trust setting:

- `balanced` is the default. Live turns may ask the user to confirm high-risk
  categories until Atagia has enough category-level consent signal. Backfills
  and admin imports route sensitive candidates to admin review.
- `trusted_private` is explicit broad consent from the user. Atagia does not
  create pending confirmations or review-required memories solely because a
  candidate is sensitive, private, or from a backfill/import path. Low
  confidence or other non-consent quality failures may still require review.

Hosts may either pass `memory_privacy_mode` on sidecar calls or store it with
Atagia's user memory preferences:

- `GET /v1/users/{user_id}/memory-preferences`
- `PUT /v1/users/{user_id}/memory-preferences`

`PUT` accepts `remember_across_chats`, `remember_across_devices`, and
`memory_privacy_mode`.

`SidecarBridge` remains fail-open for chat runtime paths. When an operation
returns `False` or `None`, `bridge.last_error` contains a `SidecarBridgeError`
with `operation`, `error_type`, `message`, optional HTTP `status_code`, and
optional parsed `details` for admin/sync logs.

## `connect_atagia` Client Facade

`connect_atagia` is the simpler alternative to `SidecarBridge` for hosts that
want one client surface to cover both in-process and remote-service Atagia
deployments. It returns an `AtagiaClient` whose async methods (`create_user`,
`create_conversation`, `get_context`, `chat`, `add_response`, `ingest_message`,
`flush`, `get_processing_status`, the memory-confirmation and admin
review helpers, `get_worker_control`, `set_worker_control`, `close`) behave
identically regardless of transport. The same host code path works whether
Atagia runs in the same Python process or behind an HTTP service.

Use `connect_atagia` when the host wants transport flexibility and is willing
to handle errors directly. Use `SidecarBridge` when the host wants the
fail-open chat-runtime contract (operations return `False`/`None` on error and
populate `bridge.last_error`) and the Aurvek-specific helpers.

Three transports are supported:

- `local` wraps an in-process `Atagia` engine and uses SQLite directly.
- `http` talks to a remote Atagia service over REST.
- `auto` (default) picks `http` when a `base_url` is provided (or
  `ATAGIA_BASE_URL` is set in the environment), otherwise falls back to
  `local`.

In-process example:

```python
from atagia.client import connect_atagia

client = await connect_atagia(
    transport="local",
    db_path="/var/lib/atagia/atagia.db",
    redis_url="redis://localhost:6379/0",
    anthropic_api_key="sk-ant-...",
)
async with client:
    await client.create_user("user-123")
    conversation_id = await client.create_conversation(
        user_id="user-123",
        conversation_id=None,
        workspace_id="workspace-1",
        space_id="workspace-1",
        platform_id="web",
    )
    context = await client.get_context(
        user_id="user-123",
        conversation_id=conversation_id,
        message="What did we decide last week?",
    )
```

HTTP service example:

```python
from atagia.client import connect_atagia

client = await connect_atagia(
    transport="http",
    base_url="https://atagia.example.com",
    api_key="svc-...",
    admin_api_key="adm-...",
    timeout=30.0,
)
async with client:
    result = await client.chat(
        user_id="user-123",
        conversation_id="conv-42",
        message="Summarize my pending tasks.",
    )
```

Auto-mode selection rule: if `base_url` is passed or `ATAGIA_BASE_URL` is set,
`connect_atagia` returns the HTTP client; otherwise it returns the local
client backed by `db_path` (resolved from `ATAGIA_DB_PATH`,
`ATAGIA_SQLITE_PATH`, or `atagia.db` in that order). Environment variables
read by auto mode:

- `ATAGIA_BASE_URL` -- HTTP service base URL
- `ATAGIA_SERVICE_API_KEY` -- per-user HTTP API key
- `ATAGIA_ADMIN_API_KEY` -- admin HTTP API key for worker-control and review
- `ATAGIA_DB_PATH` or `ATAGIA_SQLITE_PATH` -- local SQLite database path

## Confirmation And Review

Host user settings can list and resolve true live-turn pending confirmations:

- `GET /v1/users/{user_id}/memory-confirmations`
- `POST /v1/users/{user_id}/memory-confirmations/{memory_id}/confirm`
- `POST /v1/users/{user_id}/memory-confirmations/{memory_id}/decline`
- Bridge helpers: `list_pending_memory_confirmations`,
  `confirm_pending_memory`, `decline_pending_memory`

Admin tooling can inspect and remove review-required memories without writing
Atagia tables directly:

- `GET /v1/admin/memory-review`
- `POST /v1/admin/memory-review/{user_id}/{memory_id}/archive`
- `POST /v1/admin/memory-review/{user_id}/{memory_id}/delete`
- Bridge helpers: `list_review_required_memories`,
  `archive_review_required_memory`, `delete_review_required_memory`

## Processing Stop Switch

Host applications can ask Atagia to pause background memory work before a
restart, backup restore, or controlled maintenance window:

```python
await bridge.pause_new_jobs(reason="host backup")
await bridge.drain_and_pause(reason="restart", timeout_seconds=30)
await bridge.hard_pause(reason="restore database")
await bridge.resume_processing(reason="maintenance complete")
```

Modes:

- `pause_new_jobs`: Atagia still stores incoming host messages, but does not
  enqueue new message-derived memory jobs.
- `drain_and_pause`: Atagia first blocks new message-derived jobs, then waits
  for already queued work to drain, and remains paused.
- `hard_pause`: Atagia workers stop claiming any further stream jobs. In-flight
  jobs are allowed to finish.
- `active`: normal processing.

HTTP hosts need an admin key for this route. Configure `ATAGIA_ADMIN_API_KEY`
or pass `admin_api_key` to `connect_atagia()` / `SidecarBridgeConfig`. The
underlying admin endpoint is `GET/POST /v1/admin/worker-control`.

Atagia can also pause itself. The worker circuit breaker watches recent
background job outcomes and switches to `hard_pause` when failures dominate the
window. Defaults: enabled, 20 failures, 180 seconds, minimum 0.8 failure ratio.
This covers provider balance/auth failures, provider outages, local network
loss, and broad worker error storms without relying on provider-specific error
text. Configure with:

- `ATAGIA_WORKER_CIRCUIT_BREAKER_ENABLED`
- `ATAGIA_WORKER_CIRCUIT_BREAKER_FAILURE_THRESHOLD`
- `ATAGIA_WORKER_CIRCUIT_BREAKER_WINDOW_SECONDS`
- `ATAGIA_WORKER_CIRCUIT_BREAKER_MIN_FAILURE_RATIO`
