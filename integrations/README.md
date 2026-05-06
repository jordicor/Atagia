# Atagia Integrations

This directory contains reference adapters, host-specific notes, and copyable
integration scaffolds for platforms that want to use Atagia as a memory
sidecar. It is not a runtime package.

Reusable Python code belongs under `src/atagia/integrations/` so it ships with
the package. This top-level directory is for platform glue, examples, and
operator notes that a host application can copy or adapt.

Production host code should import the canonical contract from
`atagia.integrations`, not from this directory. Platform folders here should
stay thin examples over that package API so there is no second bridge contract.

## Current Layout

| Path | Status | Purpose |
|---|---:|---|
| `src/atagia/integrations/sidecar_bridge.py` | Working scaffold | Generic fail-open Python bridge over local or HTTP Atagia transports |
| `src/atagia/integrations/message_projection.py` | Working scaffold | Safe conversion of common host/provider message shapes into text |
| `src/atagia/integrations/prompt_injection.py` | Working scaffold | Prompt injection helpers for host-managed LLM calls |
| `src/atagia/integrations/aurvek.py` | Working conventions | Aurvek ID helpers; no Aurvek imports or runtime dependency |
| `integrations/aurvek/` | Copyable example | Aurvek-style host wrapper over the canonical package bridge |
| `integrations/openai-compatible/` | MVP implemented | Universal proxy surface for hosts that only speak OpenAI-compatible APIs |
| `integrations/sillytavern/` | Scaffolded | SillyTavern proxy setup plus native browser extension scaffold |
| `integrations/open-webui/` | Scaffolded | Copyable Open WebUI filter function |
| `integrations/openclaw/` | Scaffolded | Copyable OpenClaw pre/post model adapter shape |
| `integrations/hermes/` | Scaffolded | Copyable Hermes/Honcho-style memory provider facade |

## Minimum Host Pattern

Every host adapter should do the same small loop:

1. Map the verified host `user_id` and `conversation_id` to stable Atagia IDs.
2. Call `ensure_user_and_conversation()` before or during warmup.
3. Convert the current user message to text with `message_to_text()`.
4. Call `get_context_for_turn()` before the host LLM request, passing a stable
   `message_id` when the host has one. For historical imports, also pass
   `source_seq` as the host conversation-local message order and
   `ingest_origin="backfill"`.
5. Append `context.system_prompt` to the host system prompt with
   `build_injection_decision()`.
6. If Atagia is acting as primary context, avoid also sending a huge duplicated
   local history window.
7. After the host model responds, call `record_assistant_response()` with the
   host response `message_id` when available.
8. Fail open: if Atagia is disabled or unavailable, the host should keep using
   its existing context path.

`message_id` is optional and idempotent in the sidecar contract. Retrying the
same role/text with the same id does not duplicate; reusing the id for different
content returns a conflict.

`source_seq` is the canonical optional order field for backfills. When present,
Atagia stores it as the conversation `seq`, allowing retries of older failed
messages to preserve chronological order. Host examples should not introduce a
parallel `host_seq` alias.

`ingest_origin` is the canonical live-vs-import selector. The default
`live_turn` allows normal user confirmation prompts. `backfill` and
`admin_import` default to admin review for sensitive candidates, so old history
does not create pending user confirmations.

`memory_privacy_mode` is the canonical user trust selector. `balanced` is the
default and preserves confirmation/review gates for sensitive storage.
`trusted_private` is explicit broad consent from the user: Atagia will not
route a candidate to pending confirmation or review solely because it is
sensitive, private, or imported.

## What Is Still Missing For Plug-And-Play

- Live SillyTavern extension smoke validation.
- Live Open WebUI filter smoke validation and packaging.
- OpenClaw and Hermes concrete adapter APIs once their extension hooks are
  finalized against real installs.
- OpenAI-compatible proxy hardening against real OpenAI-compatible clients,
  especially tool calling, model aliases, and streaming failure recovery.
- Importers for chat histories beyond the current private Aurvek ChatLab.
- Host UI for the canonical pending-confirmation and admin review endpoints.
- A visible context/injection inspector for debugging what Atagia added.
