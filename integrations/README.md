# Atagia Integrations

This directory contains reference adapters, host-specific notes, and copyable
integration scaffolds for platforms that want to use Atagia as a memory
sidecar.

Reusable Python code belongs under `src/atagia/integrations/` so it ships with
the package. This top-level directory is for platform glue, examples, and
operator notes that a host application can copy or adapt.

## Current Layout

| Path | Status | Purpose |
|---|---:|---|
| `src/atagia/integrations/sidecar_bridge.py` | Working scaffold | Generic fail-open Python bridge over local or HTTP Atagia transports |
| `src/atagia/integrations/message_projection.py` | Working scaffold | Safe conversion of common host/provider message shapes into text |
| `src/atagia/integrations/prompt_injection.py` | Working scaffold | Prompt injection helpers for host-managed LLM calls |
| `integrations/aurvek/` | Reference adapter | Aurvek-style integration based on the live private bridge |
| `integrations/openai-compatible/` | Planned | Universal proxy surface for hosts that only speak OpenAI-compatible APIs |
| `integrations/sillytavern/` | Planned | SillyTavern extension or proxy setup notes |
| `integrations/open-webui/` | Planned | Open WebUI function/filter integration notes |
| `integrations/openclaw/` | Planned | OpenClaw memory-provider integration notes |
| `integrations/hermes/` | Planned | Hermes/Honcho-style memory integration notes |

## Minimum Host Pattern

Every host adapter should do the same small loop:

1. Map the verified host `user_id` and `conversation_id` to stable Atagia IDs.
2. Call `ensure_user_and_conversation()` before or during warmup.
3. Convert the current user message to text with `message_to_text()`.
4. Call `get_context_for_turn()` before the host LLM request.
5. Append `context.system_prompt` to the host system prompt with
   `build_injection_decision()`.
6. If Atagia is acting as primary context, avoid also sending a huge duplicated
   local history window.
7. After the host model responds, call `record_assistant_response()`.
8. Fail open: if Atagia is disabled or unavailable, the host should keep using
   its existing context path.

## What Is Still Missing For Plug-And-Play

- An OpenAI-compatible memory proxy (`/v1/chat/completions` with streaming) so
  prompt-only or API-configurable apps can use Atagia without a Python adapter.
- Native SillyTavern extension packaging and UI.
- Open WebUI function/filter packaging.
- OpenClaw and Hermes concrete adapter APIs once their extension hooks are
  finalized against real installs.
- Importers for chat histories beyond the current private Aurvek ChatLab.
- A memory review/edit UI suitable for end users.
- A visible context/injection inspector for debugging what Atagia added.
