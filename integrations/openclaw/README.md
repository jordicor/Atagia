# OpenClaw Integration

Status: implemented, mock-verified, live smoke pending.

OpenClaw should remain the source of truth for live agent state, tools, and
session lifecycle. Atagia provides retrieved continuity context and long-horizon
memory.

## Files

- `plugin/` is the copyable JS plugin bundle.
- `atagia_adapter.py` is a Python facade over `SidecarBridge` for Python-hosted
  experiments or future wrappers.
- `LOCAL_SMOKE_RUNBOOK.md` records the current local OpenClaw install,
  OpenAI Codex auth state, and the planned live smoke procedure.

## Plugin Hooks

The JS bundle implements:

- `before_prompt_build`: fetches Atagia context and returns an event with an
  injected system prompt block.
- `llm_output`: records the assistant response.
- `before_compaction`: optionally backfills transcript messages before host
  compaction.
- `session_end`: optionally backfills transcript messages at session end.

All hooks fail open by default and update `get_status()` with the last Atagia
request, resolved IDs, injected preview, and error.

## Importer

Use `integrations/importers/atagia_importers.py` for offline OpenClaw session
transcripts shaped as `messages`, `transcript`, or `sessionFile.messages`.

## Smoke Checklist

- Plugin loads in a real OpenClaw install.
- `before_prompt_build` receives the current user message and returns an Atagia
  memory block.
- `llm_output` records the assistant response once.
- Atagia down/API error leaves OpenClaw generation working.
- `before_compaction` or `session_end` can backfill a transcript rerunnably.

For the current local test plan, start with `LOCAL_SMOKE_RUNBOOK.md`.
