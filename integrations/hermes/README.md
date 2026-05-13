# Hermes Integration

Status: implemented, mock-verified, live smoke pending.

Hermes-style stacks often expose a memory-provider abstraction. Atagia now ships
a copyable provider bundle at `plugins/memory/atagia/`, plus the older
`atagia_provider.py` facade over `SidecarBridge` for direct Python integration.

## Provider Bundle

`plugins/memory/atagia/` implements a minimal `MemoryProvider`:

- `is_available`
- `initialize`
- `get_config_schema`
- `save_config`
- `prefetch`
- `sync_turn`
- `on_session_end`
- `on_memory_write` no-op
- `shutdown`

`sync_turn` and `on_session_end` enqueue work on a daemon thread, so Hermes
generation does not block on Atagia persistence.

`on_memory_write` intentionally does nothing for now. Curated Hermes memory
records should not be converted into fake chat turns until there is a clear
semantic mapping.

## Importer

Use `integrations/importers/atagia_importers.py` for Hermes exports shaped as
`messages`, `transcript`, or `memories` with text fields.

## Smoke Checklist

- Provider imports against a real Hermes install and sees
  `agent.memory_provider.MemoryProvider`.
- `initialize(config)` returns available with the local Atagia URL.
- `prefetch()` returns a `system_prompt` and stores the user message.
- `sync_turn()` queues assistant persistence without blocking generation.
- `on_session_end()` can backfill session transcript messages rerunnably.
- `on_memory_write()` remains a documented no-op.
- Atagia down/API error fails open.
