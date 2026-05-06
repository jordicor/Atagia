# OpenClaw Integration

Status: adapter shape scaffolded; hook validation pending.

OpenClaw should be treated as a host with its own memory and agent lifecycle
concepts. Atagia should not replace its application state; it should provide
retrieved continuity context and long-horizon memory.

## Target Shape

- Map OpenClaw's user/session/agent IDs to Atagia IDs.
- Use `SidecarBridge` before model calls.
- Pass structured attachments or artifacts when OpenClaw exposes them.
- Record assistant responses after model calls.
- Keep Atagia memory advisory: OpenClaw remains the source of truth for live
  task state and tool execution state.

## Missing Atagia Pieces

- Concrete OpenClaw extension hook validation.
- Adapter package or sample plugin around `atagia_adapter.py` once a real hook
  contract is confirmed.
- Importer for existing OpenClaw memory files, if the host exposes them.

## Files

- `atagia_adapter.py` is a copyable pre-model/post-model adapter shape. It
  intentionally avoids OpenClaw imports until the target install's hook API is
  validated.
