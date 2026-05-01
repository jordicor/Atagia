# Hermes Integration

Status: planned.

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
- Provider package or plugin scaffold.
- Import/export mapping between Hermes memory records and Atagia objects.
