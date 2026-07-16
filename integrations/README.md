# Atagia Integrations

This directory contains reference adapters, host-specific notes, and copyable
integration bundles for platforms that use Atagia as a memory sidecar. It is
not a runtime package.

Reusable package code belongs under `src/atagia/integrations/`. Platform folders
here stay thin over the canonical `SidecarService` / `SidecarBridge` contract so
there is no second integration API.

## Readiness Matrix

| Platform | implemented | contract-verified | deployed-smoke-pending | importer-ready | review-ui-ready | Notes |
|---|---:|---:|---:|---:|---:|---|
| OpenAI-compatible proxy | yes | yes | yes | n/a | n/a | Headers and `metadata` identity, streaming SSE, tool calls, usage chunks, fail-open persistence |
| SillyTavern extension | yes | yes | yes | yes | partial | Browser extension uses `chatMetadata`, `setExtensionPrompt`, stable IDs, response persistence, debug inspector |
| Open WebUI filter | yes | yes | yes | n/a | partial | `inlet()` context injection, `outlet()` response persistence, debug state; API-direct users should prefer the proxy |
| OpenClaw plugin | yes | yes | yes | yes | partial | Pinned native JS plugin with `before_prompt_build`, `agent_end`, `before_compaction`, `session_end` |
| Hermes MemoryProvider | yes | yes | yes | yes | partial | Requires the versioned downstream patch pinned to Hermes 0.18.2; vanilla host is rejected. Stable SessionDB IDs and pre-prefetch selection/cutoff signals cover retry, undo, and regeneration. |

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
| `integrations/sillytavern/extension/` | pinned contract verified | Browser extension paired with the server-side secret boundary; live-host turn smoke pending before production support |
| `integrations/open-webui/` | pinned contract verified | Copyable Open WebUI Filter Function with bounded per-request state |
| `integrations/openclaw/plugin/` | official loader/runtime verified | Copyable OpenClaw plugin bundle pinned to the declared host commit; live Gateway/model turn pending |
| `integrations/hermes/plugins/memory/atagia/` | pinned patched-host ABI/loader verified | Hermes MemoryProvider with the required `hermes.memory-selection.v1` downstream host patch; live host smoke pending |
| `integrations/importers/` | contract verified | Offline importers for chat/session exports |

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

`source_seq` is the optional conversation-local order field. It is derived only
from a stable host ordinal, never from message text. A retry of the same host
event preserves it; a swipe, continuation, or regeneration that reuses a host
position uses the host's explicit generation identity and omits `source_seq`
when the host cannot provide a distinct monotonic ordinal. This is the common
adapter requirement, not a claim that every vanilla host exposes the necessary
signal. Atagia's Hermes integration therefore ships a downstream patch pinned
to the exact 0.18.2 commit; unpatched Hermes is rejected before provider effects.

## Verification

The pinned contract/loader gate for these bundles is:

```bash
./.venv/bin/pytest \
  tests/api/test_openai_proxy.py \
  tests/integrations/test_platform_scaffolds.py \
  tests/integrations/test_open_webui_filter.py \
  tests/integrations/test_platform_importers.py \
  tests/integrations/test_hermes_plugin.py \
  tests/integrations/test_node_bundles.py \
  tests/integrations/test_openclaw_plugin.py \
  tests/integrations/test_sillytavern_secret_boundary.py
```

Deployment-specific live turns remain separate operational gates for
SillyTavern, OpenClaw, and Hermes; their READMEs name the exact remaining work
without weakening the versioned local contract evidence above.
