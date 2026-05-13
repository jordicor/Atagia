# Atagia Hermes Memory Provider

Status: implemented, mock-verified, live smoke pending.

Copy `plugins/memory/atagia/` into a Hermes plugin path and configure the
provider as the session memory provider.

## Provider Methods

- `is_available()` checks whether the provider is enabled and has a base URL.
- `initialize(config)` and `save_config(config)` load runtime configuration.
- `get_config_schema()` returns a simple JSON schema for UI/config editors.
- `prefetch(query, **kwargs)` fetches Atagia context and returns
  `{"system_prompt": ...}`.
- `sync_turn(turn, **kwargs)` queues non-blocking user/assistant persistence on a
  daemon thread.
- `on_session_end(session, **kwargs)` queues transcript backfill with
  `ingest_origin="backfill"` and `confirmation_strategy="admin_review_only"`.
- `on_memory_write(...)` is intentionally a no-op. Curated Hermes memories are
  not converted into synthetic chat turns in this version.
- `shutdown()` stops the daemon worker.

## Environment

```bash
ATAGIA_BASE_URL=http://127.0.0.1:8100
ATAGIA_SERVICE_API_KEY=change-me
```

Atagia stores conversational memory and returns prompt context. Hermes remains
the source of truth for tasks, tools, session state, and curated memory records.
