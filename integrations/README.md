# Atagia Integrations

This directory contains reference adapters, host-specific notes, and copyable
integration bundles for platforms that use Atagia as a memory sidecar. It is
not a runtime package.

Reusable package code belongs under `src/atagia/integrations/`. Platform folders
here stay thin over the canonical `SidecarService` / `SidecarBridge` contract so
there is no second integration API.

## Readiness Matrix

| Platform | implemented | mock-verified | live-smoke-pending | importer-ready | review-ui-ready | Notes |
|---|---:|---:|---:|---:|---:|---|
| OpenAI-compatible proxy | yes | yes | yes | n/a | n/a | Headers and `metadata` identity, streaming SSE, tool calls, usage chunks, fail-open persistence |
| SillyTavern extension | yes | yes | yes | yes | partial | Browser extension uses `chatMetadata`, `setExtensionPrompt`, stable IDs, response persistence, debug inspector |
| Open WebUI filter | yes | yes | yes | n/a | partial | `inlet()` context injection, `outlet()` response persistence, debug state; API-direct users should prefer the proxy |
| OpenClaw plugin | yes | yes | yes | yes | partial | Copyable JS plugin with `before_prompt_build`, `llm_output`, `before_compaction`, `session_end` |
| Hermes MemoryProvider | yes | yes | yes | yes | partial | Copyable `plugins/memory/atagia/` provider with non-blocking daemon sync worker |

`partial` review UI means the bundle exposes a mini inspector/debug status for
the last Atagia request, resolved IDs, injected preview, and fail-open errors.
Full memory review/edit UX still belongs to the host-specific live smoke pass.

## Current Layout

| Path | Status | Purpose |
|---|---:|---|
| `src/atagia/integrations/sidecar_bridge.py` | implemented | Generic fail-open Python bridge over local or HTTP Atagia transports |
| `src/atagia/integrations/message_projection.py` | implemented | Safe conversion of common host/provider message shapes into text |
| `src/atagia/integrations/prompt_injection.py` | implemented | Prompt injection helpers for host-managed LLM calls |
| `src/atagia/integrations/aurvek.py` | implemented | Aurvek ID helpers; no Aurvek imports or runtime dependency |
| `integrations/aurvek/` | mock-verified | Aurvek-style host wrapper over the canonical package bridge |
| `integrations/openai-compatible/` | mock-verified | Universal OpenAI-compatible proxy surface |
| `integrations/sillytavern/extension/` | mock-verified | Copyable SillyTavern browser extension |
| `integrations/open-webui/` | mock-verified | Copyable Open WebUI Filter Function |
| `integrations/openclaw/plugin/` | mock-verified | Copyable OpenClaw plugin bundle |
| `integrations/hermes/plugins/memory/atagia/` | mock-verified | Copyable Hermes MemoryProvider plugin |
| `integrations/importers/` | mock-verified | Offline importers for chat/session exports |

## Common Contract

Every host adapter should follow the same loop:

1. Map verified host `user_id`, `platform_id`, and `conversation_id` to stable
   Atagia IDs. The proxy rejects requests missing any of these three.
2. Pass stable `message_id` and `source_seq` for the live user turn when the host
   can derive them.
3. Call context retrieval before the host LLM request using
   `ingest_origin="live_turn"` and
   `confirmation_strategy="live_prompt_allowed"`.
4. Inject only the returned prompt context. Do not persist synthetic Atagia
   prompt blocks into host chat history.
5. After the model responds, persist the assistant response with its own stable
   `response_message_id` / `response_source_seq` when available.
6. For imports/backfills, call `/v1/conversations/{id}/messages` with
   `ingest_origin="backfill"` and
   `confirmation_strategy="admin_review_only"`.
7. Pass `memory_privacy_mode` explicitly. Use `balanced` by default and
   `trusted_private` only when the user has granted broad private-memory trust.
8. Fail open: if Atagia is disabled or unavailable, the host continues its
   normal context path.

`message_id` is idempotent. Retrying the same role/text with the same ID does
not duplicate; reusing an ID for different content returns a conflict.

`source_seq` is the optional conversation-local order field. These bundles use
deterministic source sequences that remain rerunnable while changing for edited
or regenerated text.

## Verification

The mock verification gate for these bundles is:

```bash
./.venv/bin/pytest \
  tests/api/test_openai_proxy.py \
  tests/integrations/test_platform_scaffolds.py \
  tests/integrations/test_open_webui_filter.py \
  tests/integrations/test_platform_importers.py \
  tests/integrations/test_hermes_plugin.py \
  tests/integrations/test_node_bundles.py
```

The next session should run live smoke tests against real SillyTavern,
OpenClaw, and Hermes installs.
