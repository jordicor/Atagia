# Hermes Integration

Status: provider shape scaffolded; concrete API validation pending.

Hermes-style stacks often already have a memory-provider abstraction. Atagia can
fit either as a provider adapter or as an upstream sidecar proxy.

## Target Shape

- If Hermes exposes a memory provider API, implement an Atagia provider that
  delegates to `SidecarBridge`.
- If Hermes only exposes model-provider configuration, use the planned
  OpenAI-compatible memory proxy.
- Preserve Hermes' own session/task state as host state; use Atagia for
  long-horizon relationship, preference, belief, and project memory.

## Missing Atagia Pieces

- Concrete Hermes adapter API validation against a real install.
- Provider package or plugin wrapper around `atagia_provider.py` once the target
  provider interface is confirmed.
- Import/export mapping between Hermes memory records and Atagia objects.

## Files

- `atagia_provider.py` is a copyable retrieve/record/ingest facade over
  `SidecarBridge`. Wrap it in the host's concrete memory-provider interface.
