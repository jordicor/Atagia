# Atagia Memory Plugin For OpenClaw

Status: implemented, mock-verified, live smoke pending.

Copy this `plugin/` directory into an OpenClaw plugin location and configure the
plugin with the local Atagia sidecar URL and service key.

## Hooks

- `before_prompt_build` fetches Atagia context with a stable user message ID and
  returns an event with an injected system prompt block.
- `llm_output` records the assistant response with a stable response ID.
- `before_compaction` and `session_end` optionally backfill a transcript using
  `ingest_origin="backfill"` and `confirmation_strategy="admin_review_only"`.

All hooks fail open by default. Inspect `get_status()` for the last request,
resolved IDs, injected preview, and fail-open error.

## Environment

```bash
ATAGIA_BASE_URL=http://127.0.0.1:8100
ATAGIA_SERVICE_API_KEY=change-me
```

OpenClaw remains the source of truth for session state, tools, and compaction.
Atagia only stores conversational memory and returns advisory context.
