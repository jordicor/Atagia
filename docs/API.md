# Atagia REST API Reference

This document lists every FastAPI route registered by the Atagia service. It
is the canonical reference linked from the [README](../README.md). For request
and response schemas, consult the live `/openapi.json` produced by the running
service.

All routes are mounted under the `/v1` prefix. Admin routes are mounted under
`/v1/admin`. There is no unversioned alias.

---

## Authentication

Atagia runs in one of two modes, selected by the `ATAGIA_SERVICE_MODE`
environment variable.

- **Library mode** (`ATAGIA_SERVICE_MODE=false`): the caller is trusted. The
  `user_id` provided in each request body or query is taken at face value.
  No `Authorization` header is required.
- **Service mode** (`ATAGIA_SERVICE_MODE=true`): every route requires an
  `Authorization: Bearer <token>` header.

Two API keys are recognized in service mode:

| Key | Used by | Required by |
|---|---|---|
| `ATAGIA_SERVICE_API_KEY` | Shared server-side clients | All non-admin routes below |
| `ATAGIA_ADMIN_API_KEY` | Admin / operational routes | All `/v1/admin/*` routes |

`ATAGIA_SERVICE_API_KEY` is one shared non-admin service credential, not a
per-user token. Keep it in trusted server-side components and never expose it
to browser code or another untrusted client. Service mode separately requires
the trusted `X-Atagia-User-Id` header on user-facing routes. The handler rejects
any request whose path or body `user_id` does not match that asserted identity.
Admin routes do not require the user-id header because they operate across
users by design.

The OpenAI-compatible proxy (`/v1/models`, `/v1/chat/completions`) uses the
same `ATAGIA_SERVICE_API_KEY` but emits OpenAI-shaped error envelopes
instead of FastAPI's default error format.

Ordinary chat, sidecar, and proxy requests always run with server-derived
standard authority and privacy enforcement enabled. Request bodies, metadata,
and headers cannot select a privilege level, master authority, or a different
privacy-enforcement mode. The admin API key is valid only on `/v1/admin/*`; it
does not grant master authority on ordinary routes. Trusted library and
evaluation callers can still set these internal controls without using HTTP.

---

## Core routes

All core routes share the `/v1` prefix and authenticate with
`ATAGIA_SERVICE_API_KEY` in service mode.

### Users

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/users` | Create a user record, or return the existing one if it already exists. |
| `POST` | `/v1/users/{user_id}/erase` | Hard-erase all data for a user under a confirmation token, returning an `ErasureReport`. |
| `GET` | `/v1/users/{user_id}/memory-preferences` | Return the user's cross-chat, cross-device, and privacy-mode memory preferences. |
| `PUT` | `/v1/users/{user_id}/memory-preferences` | Update the user's memory preferences. |
| `GET` | `/v1/users/{user_id}/memory-confirmations` | List memory candidates pending user confirmation, with optional filters by conversation, platform, persona, character, and category. |
| `POST` | `/v1/users/{user_id}/memory-confirmations/{memory_id}/confirm` | Confirm a pending memory candidate so it becomes a regular memory object. |
| `POST` | `/v1/users/{user_id}/memory-confirmations/{memory_id}/decline` | Decline a pending memory candidate. |
| `GET` | `/v1/users/{user_id}/processing-status` | Return the worker job status for a user across the configured namespace filters. |
| `GET` | `/v1/users/{user_id}/contract` | Return the current interaction contract dimensions projected for the user under the given namespace. |
| `GET` | `/v1/users/{user_id}/state` | Return a snapshot of the user's memory state under the given namespace. |

### Workspaces

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/workspaces` | Create a workspace for a user, or return the existing one if `workspace_id` is supplied and already exists. |

### Conversations

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/conversations` | Create or look up a conversation in its declared coordinates (workspace, mode, persona, platform, character, mind, embodiment, realm, space). |
| `POST` | `/v1/conversations/{conversation_id}/incognito` | Set or clear the incognito flag on a conversation. |
| `POST` | `/v1/conversations/{conversation_id}/save-from-incognito` | Prepare a review payload listing memories that would be saved if the conversation exits incognito. |
| `POST` | `/v1/conversations/{conversation_id}/close` | Close a conversation, optionally purging it, gated by a confirmation token. |
| `POST` | `/v1/conversations/{conversation_id}/archive` | Archive a conversation without deleting its content. |
| `POST` | `/v1/conversations/{conversation_id}/delete` | Delete a conversation under a confirmation token, returning a `DeletionReport`. |
| `GET` | `/v1/conversations/{conversation_id}/processing-status` | Return the worker job status for a specific conversation. |

### Chat and sidecar

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/{conversation_id}/reply` | Run the full retrieval-and-generation pipeline and return the assistant reply along with retrieval and memory-processing metadata. |
| `POST` | `/v1/conversations/{conversation_id}/context` | Run retrieval only and return the assembled `ContextResult` without generating a reply, for hosts that own their own LLM call. |
| `POST` | `/v1/conversations/{conversation_id}/messages` | Sidecar-mode ingest of a single message produced by the host, idempotent on `message_id` and `source_seq`. |
| `POST` | `/v1/conversations/{conversation_id}/responses` | Sidecar-mode ingest of an assistant response produced by the host's own LLM, idempotent on `message_id` and `source_seq`. |
| `POST` | `/v1/flush` | Drain pending background work for the caller and return the updated processing status, blocking up to `timeout_seconds`. |

### Activity and warm-up

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/users/{user_id}/activity/conversations` | Return the ranked list of "hot" conversations for a user with optional limit, workspace, mode, and persona filters. |
| `POST` | `/v1/conversations/{conversation_id}/warmup` | Pre-warm a specific conversation by hydrating its activity stats and recent context. |
| `POST` | `/v1/users/{user_id}/warmup` | Pre-warm the user's recommended conversations within message budgets. |

### Memory objects and feedback

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/memory/feedback` | Record user feedback on a memory that was surfaced inside a specific retrieval event. |
| `GET` | `/v1/memory/objects/{memory_id}` | Return a single memory object visible under the requested namespace, scrubbed of internal fields. |
| `PATCH` | `/v1/memories/{memory_id}` | Edit the canonical text of an existing memory object owned by the user. |
| `POST` | `/v1/memories/{memory_id}/delete` | Delete or hard-delete a memory object under a confirmation token, returning a `DeletionReport`. |

### Verbatim pins

Mounted under `/v1/verbatim-pins`. These let the user pin exact strings that
must be reproduced verbatim by the assistant.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/verbatim-pins` | Create a verbatim pin scoped to the requested coordinates. |
| `GET` | `/v1/verbatim-pins` | List the user's verbatim pins with optional status, scope, target-kind, target-id, deletion, and active-only filters. |
| `GET` | `/v1/verbatim-pins/{pin_id}` | Return a single verbatim pin by id. |
| `PATCH` | `/v1/verbatim-pins/{pin_id}` | Update one or more fields on a verbatim pin. |
| `DELETE` | `/v1/verbatim-pins/{pin_id}` | Soft-delete a verbatim pin and return its final record. |

### OpenAI-compatible proxy

Hosts that already speak the OpenAI Chat Completions wire format can call
Atagia as a drop-in. The proxy reads coordinates from `X-Atagia-*` headers
(conversation, mode, workspace, persona, platform, character, incognito,
cross-chat memory, message ids, source sequences, ingest origin,
confirmation strategy, memory privacy mode) and applies the standard
retrieval pipeline before delegating to the underlying LLM. Errors are
returned in the OpenAI error envelope.

`max_tokens` and `max_completion_tokens` are external answer ceilings. When
both are present, Atagia uses the smaller; the server also caps the result with
`ATAGIA_PROXY_MAX_OUTPUT_TOKENS`. A positive client ceiling is never increased
to an internal structured-output floor. Provider-supplied usage is returned
unchanged when available (and, for streams, only when
`stream_options.include_usage=true`); absent usage is omitted rather than
estimated. Provider stop reasons are normalized to `stop`, `tool_calls`,
`length`, or `error` where applicable.

String message content is always literal, even when it looks like a JSON
array or object. Only content that arrived as typed structured blocks takes
the multimodal/tool projection path.

`message_id` and `response_message_id` form an all-or-none proxy-turn pair.
Supplying one without the other returns `400` before retrieval, mutation, or
provider execution. A source sequence is valid only with its corresponding
message ID, while request and response sequences are independently optional.
With neither ID, each HTTP request creates a new turn. With both IDs, an
identical completed retry replays the stored normalized response without
another context or provider call; streaming and non-streaming framing may be
interchanged. Response-determining request changes, shared IDs, swapped roles,
or cross-namespace reuse return a stable `409` conflict. A live compatible
request returns `request_in_progress` with `Retry-After`; an interrupted stream
that exposed bytes requires a new pair and returns
`stream_retry_requires_new_ids` on reuse.

Tool-only assistant turns and trailing tool-result batches are stored as
lossless ordered structured metadata with stable call IDs and causal links.
The proxy atomically commits the response, replay envelope, reciprocal links,
and every required durable post-response root job before returning a buffered
success or emitting a streaming finish/usage/`[DONE]` event. Transient context
infrastructure failures may fail open without memory, but terminal transcript
persistence does not.

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | List the Atagia-routable models exposed by the proxy. |
| `POST` | `/v1/chat/completions` | Run a chat completion through Atagia, supporting both buffered and streaming (`text/event-stream`) responses. |

---

## Admin routes

All admin routes share the `/v1/admin` prefix and authenticate with
`ATAGIA_ADMIN_API_KEY` in service mode. Routes with an explicit audited action
write to the existing admin audit log; audit coverage is route-specific rather
than implicit for every admin read. There are 29 admin endpoints in total.

### Worker control

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/worker-control` | Return the current worker-control mode and the source-job, claim, and periodic-work flags it implies. |
| `POST` | `/v1/admin/worker-control` | Set the worker-control mode, optionally draining in-flight work for `drain_and_pause`. |

### Memory review

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/memory-review` | List memory candidates in `review_required` status with optional namespace and category filters. |
| `POST` | `/v1/admin/memory-review/{user_id}/{memory_id}/archive` | Archive a review-required memory (soft-delete) under the admin actor. |
| `POST` | `/v1/admin/memory-review/{user_id}/{memory_id}/delete` | Hard-delete a review-required memory using the built-in admin confirmation token. |

The sensitive `GET /v1/admin/memory-review` operation audits successful,
denied, validation-failed, and repository-failed attempts with the actor,
normalized filters, result count, status, HTTP status, and error class. It
never copies canonical memory text, payloads, raw source content, or API keys
into the audit row. A successful sensitive read fails closed if its audit row
cannot be persisted.

### Rebuild

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/admin/rebuild/conversation/{conversation_id}` | Re-extract memories and re-project the contract for one conversation. |
| `POST` | `/v1/admin/rebuild/user/{user_id}` | Re-extract memories and re-project the contract across every conversation owned by a user. |

### Compaction and reindex

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/admin/compact/conversation/{conversation_id}` | Generate conversation-level summary chunks for a single conversation. |
| `POST` | `/v1/admin/compact/workspace/{workspace_id}` | Generate a workspace-level rollup summary. |
| `POST` | `/v1/admin/reindex` | Rebuild the FTS5 indexes for messages and memory objects. |

### Lifecycle

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/admin/lifecycle/run` | Run one full lifecycle cycle (decay, archival, retention) with an optional `dry_run` flag. |

### Metrics

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/metrics/latest` | Return the latest stored value for every metric, optionally filtered by user and assistant mode. |
| `GET` | `/v1/admin/metrics/{metric_name}/history` | Return the time-bucketed history of a named metric. |
| `POST` | `/v1/admin/metrics/compute` | Recompute a list of metrics for a given time bucket; `ccr` is queued as a worker job, the rest run inline. |
| `GET` | `/v1/admin/metrics/retrieval-summary` | Return retrieval-event aggregate statistics between two time buckets for a user or globally, split per turn surface and optionally filtered to one via `turn_surface`. Latency comes back as three separate series, each with its own sample count: `avg_retrieval_stage_latency_ms` (the measured retrieval stage, over turns whose retrieval produced memory context), `avg_failed_retrieval_stage_latency_ms` (the same measured stage over turns whose retrieval failed open, kept apart so a failed attempt never describes what a working retrieval costs) and `avg_request_to_event_wall_ms` (the request-message-to-event wall gap, which also contains reply generation, available only for events written before the turn-telemetry migration). |

### Retrieval events and memory decisions

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/retrieval-events/{event_id}` | Return a full retrieval event including its context view, ranking decisions, and audit metadata. |
| `GET` | `/v1/admin/retrieval-events/{event_id}/memory-decisions/{user_id}/{memory_id}` | Return the per-memory decision row produced for a given retrieval event. |

### Consequence chains

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/consequence-chains/{user_id}` | List the consequence chains (action -> outcome -> tendency) for a user, optionally scoped by workspace. |

### Replay and grounding

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/admin/replay/event/{retrieval_event_id}` | Re-run the retrieval pipeline against a stored event under an ablation configuration. |
| `POST` | `/v1/admin/replay/conversation/{conversation_id}` | Replay retrieval over every (or up to `message_limit`) message in a conversation under an ablation configuration. |
| `POST` | `/v1/admin/grounding/{retrieval_event_id}` | Run grounding analysis over a stored retrieval event's context view. |

### Export

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/admin/export/conversation/{conversation_id}` | Export a conversation (and optionally its retrieval traces and intimacy context) under the requested anonymization mode. |

### Memory coordinates

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/memory-coordinates/{user_id}/{memory_id}` | Inspect every coordinate field attached to a memory object. |
| `POST` | `/v1/admin/memory-coordinates/{user_id}/{memory_id}/correct` | Apply an admin correction to one or more coordinate fields, invalidating the user's context cache. |
| `GET` | `/v1/admin/memory-coordinates/{user_id}/{memory_id}/corrections` | Return the audit history of coordinate corrections for a memory. |

### Embeddings backfill

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/admin/embeddings/backfill` | Backfill missing embeddings, optionally scoped to a single user, in batches with configurable inter-batch delay. |

### Coverage backfill

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/admin/memory/coverage/backfill` | Recompute missing memory coverage members, optionally scoped to one user, under the durable admin-maintenance fence. |

### LLM run guard

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/llm-run-guard` | Return the current runtime LLM run-guard snapshot (budget and health counters); returns `{"enabled": false}` when the guard is inactive. |
| `POST` | `/v1/admin/llm-run-guard/reset` | Reset the runtime LLM run-guard counters and return the post-reset snapshot. Writes an admin audit entry. |

---

## Conventions

### Error format

User-facing routes use FastAPI's default error envelope:

```json
{ "detail": "Conversation not found for user" }
```

The OpenAI-compatible proxy returns the OpenAI-shaped envelope instead:

```json
{ "error": { "message": "...", "type": "invalid_request_error", "param": null, "code": null } }
```

### Confirmation tokens

Destructive routes (`/conversations/{id}/delete`, `/memories/{id}/delete`,
`/users/{id}/erase`, `/conversations/{id}/close` when purging, and the admin
hard-delete) require a `confirmation` field whose exact value is documented
in the request schema. Missing or wrong tokens fail with `400 Bad Request`.

### Idempotency

`message_id` and `source_seq` are the idempotency keys for sidecar writes:

- `message_id` is a host-supplied stable identifier for a single message.
  Posting the same `message_id` twice on `/conversations/{id}/messages` or
  `/conversations/{id}/responses` returns the original record with
  `idempotent_replay: true`.
- `source_seq` is a monotonic per-conversation sequence number. Out-of-order
  or conflicting sequences fail with `409 Conflict` (`SourceSequenceConflictError`)
  so the host can detect drift.

Deterministic message-ID, source-sequence, and namespace/mind conflicts are
hard errors. They do not fall through to the upstream provider and do not
create an assistant response. Transient infrastructure failures follow only
the explicitly documented fail-open paths.

`POST /v1/users` and `POST /v1/conversations` are themselves idempotent on
their resource ids: re-posting with the same id returns the existing record
unchanged.

### Worker-driven operations

Memory extraction, contract projection, compaction, graph sync, and metric
computation run in background workers. Synchronous chat and context routes
return immediately with a `memory_processing` status the caller can later
poll via `/conversations/{id}/processing-status`, `/users/{id}/processing-status`,
or block on via `/v1/flush`.

### Namespace filters

User-facing routes that read or write memories require the namespace
coordinates (`conversation_id`, `platform_id`, optional `user_persona_id`,
`character_id`, `incognito`) plus any active memory-coordinate fields the host
uses (`active_presence_id`, `mind_id`, `mind_topology`, `embodiment_id`,
`realm_id`, `space_id`) so candidate selection is scoped before ranking.
Service mode additionally rejects any request without a `platform_id`.

For an existing conversation, `user_persona_id`, `platform_id`, and
`character_id` are optional identity hints: an omitted or JSON-null hint uses
the persisted conversation identity, while every supplied non-null value must
match exactly. A conflict returns the same non-disclosing `404` as an unknown
conversation. The service-mode requirement for `platform_id` remains a route
boundary requirement and does not change this comparison rule for library
callers.

`incognito` and `cross_chat_memory` use one strict request-boundary rule. The
proxy accepts their typed request fields, `atagia_*` or unprefixed metadata
keys, and the documented `X-Atagia-Incognito` and
`X-Atagia-Cross-Chat-Memory` headers. Conversation creation, chat, and sidecar
routes use their typed fields; conversation creation and chat also accept the
same metadata keys. Repeated claims must agree, and every explicit value must
be a JSON boolean or one of the strings
`true`/`false`, `1`/`0`, `yes`/`no`, or `on`/`off`. Invalid or contradictory
claims return `400`.
After validation, `incognito=true` disables cross-chat memory, while
`incognito=false` never overrides an explicit `cross_chat_memory=false`.
With both settings absent, cross-chat memory retains its documented default of
enabled.

### Request budgets

External requests are bounded before attachment decoding, SQLite writes, or
provider execution. Defaults are 32 MiB for the encoded body, 256 KiB per
message text, 16 attachments, 10 MiB per decoded attachment, 20 MiB decoded in
aggregate, and 64 KiB of compact serialized metadata. The corresponding
`ATAGIA_REQUEST_MAX_*` settings are documented in the configuration reference.
Encoded/decoded size and count overruns return `413`; invalid base64 and other
payload-structure failures return `422`. Rejected requests leave no partial
message, artifact, blob, or provider call.

### Path identifiers

The SDK and importers use Atagia's reversible path-ID encoding for route
segments. Routes decode exactly once before repository access. IDs containing
slashes, literal percent sequences such as `%2F`, spaces, Unicode, or the
reserved `__atagia_b64_` prefix therefore round-trip without percent-decoding
ambiguity; ordinary path-safe IDs remain unchanged. A malformed reserved
transport value returns `400`.

---

## Examples

Create a user (library mode):

```bash
curl -X POST http://127.0.0.1:8100/v1/users \
  -H "Content-Type: application/json" \
  -d '{"user_id": "usr_demo"}'
```

Run a chat reply (service mode):

```bash
curl -X POST http://127.0.0.1:8100/v1/chat/cnv_demo/reply \
  -H "Authorization: Bearer $ATAGIA_SERVICE_API_KEY" \
  -H "X-Atagia-User-Id: usr_demo" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "usr_demo",
    "platform_id": "host_demo",
    "message_text": "Where did we leave off yesterday?"
  }'
```
