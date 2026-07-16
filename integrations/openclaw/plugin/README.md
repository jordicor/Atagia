# Atagia Memory Plugin For OpenClaw

Status: native manifest, loader, hook lifecycle, shutdown, and reload verified
against OpenClaw `2026.5.6` at commit
`8934095c828de8d6268e0e42d8cfe6651ccf5a1b`. A real deployed Atagia/OpenClaw
smoke remains pending.

This directory is a native OpenClaw plugin package. `openclaw.plugin.json`
declares its configuration and `package.json` points both source and installed
runtime discovery at `index.js`.

## Host Contract

The plugin registers these typed hooks through `register(api)` and `api.on`:

- `before_prompt_build` sends the current prompt to Atagia `/context` and
  returns the supported `{ prependContext }` result.
- `agent_end` selects the final assistant message from the completed host turn
  and writes it to `/responses` once. A repeated completed hook is skipped after
  durable confirmation.
- `before_compaction` reconstructs the JSONL branch selected by the hook's
  active in-memory `messages`, then sends that complete branch to Atagia's
  authoritative `/selected-transcript` replacement workflow.
- `before_reset` performs the same authoritative reconciliation from the
  active messages OpenClaw exposes immediately before `/new` or `/reset` drops
  the old session.
- `session_end` safely infers a leaf only when the transcript has exactly one
  terminal path. A branched `sessionFile` without active messages or an
  explicit leaf remains ambiguous, so the plugin defers without mutation.

OpenClaw classifies `agent_end` as a raw-conversation hook. The deployment must
therefore set `plugins.entries.atagia-memory.hooks.allowConversationAccess=true`.

## Configuration

Link the package during local development:

```bash
openclaw plugins install --link /path/to/atagia/integrations/openclaw/plugin
```

Then configure the plugin under its native entry:

```json
{
  "plugins": {
    "entries": {
      "atagia-memory": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true
        },
        "config": {
          "baseUrl": "http://127.0.0.1:8100",
          "installationId": "stable-openclaw-installation",
          "hostAccountId": "stable-openclaw-account",
          "userId": "mapped-atagia-user"
        }
      }
    }
  }
}
```

The service key should normally stay in the Gateway environment:

```bash
ATAGIA_SERVICE_API_KEY=change-me
```

Environment overrides also exist for `ATAGIA_BASE_URL`,
`ATAGIA_OPENCLAW_INSTALLATION_ID`, `ATAGIA_OPENCLAW_HOST_ACCOUNT_ID`, and
`ATAGIA_OPENCLAW_USER_ID`. The installation, host-account, and mapped-user IDs
are mandatory when the plugin is enabled; no placeholder identity is used.
The default Atagia request timeout is 10 seconds, below OpenClaw `2026.5.6`'s
default 15-second `before_prompt_build` hook budget. Keep any configured
`timeoutMs` below the corresponding OpenClaw hook timeout. Selected transcript
rebuild polling is independently bounded by `selectedTranscriptWaitMs`
(20 seconds by default) with `selectedTranscriptPollIntervalMs` (250 ms by
default). An unfinished rebuild remains durable and the next authoritative hook
continues polling the same operation.

The authoritative exact pin on OpenClaw `2026.5.6` is the runtime version
check enforced during registration; a different host version fails plugin
loading until this contract and its pinned loader tests are reviewed and
updated. The install metadata (`install.minHostVersion` and the related
version fields) declares `>=2026.5.6` as the install floor, while the related
compatibility fields stay pinned to `2026.5.6`. The runtime check is what
enforces exactness.

## Identity And Ordering

Atagia message IDs are SHA-256 hashes of the canonical external identity tuple:

- integration kind (`openclaw`);
- stable host installation and account;
- mapped Atagia user;
- immutable OpenClaw transcript session (`sessionId`); `sessionKey` is routing
  metadata and never a canonical conversation identity;
- native live event (`runId`) or selected transcript entry ID;
- role and generation.

Message text is never an identity input. Live retries with the same `runId`
reuse the mapping, while later runs with identical text allocate new IDs and
monotonic conversation-scoped `source_seq` values. If a transcript entry lacks
a native ID, its selected-branch ordinal is the durable fallback.

The durable mapping store defaults to
`$OPENCLAW_STATE_DIR/plugins/atagia-memory/identity.json`. It is written before
network effects and normally contains identity/order metadata only—never
retrieved context, credentials, or raw errors. While a selected-transcript POST
has no authenticated acknowledgement, it temporarily retains that exact
request as a mode-0600 durable outbox so a reset followed by process loss cannot
discard the old session's only replay source. The payload is removed as soon as
Atagia acknowledges the operation. The store also persists sanitized scope
coordinates, the monotonic selection epoch, retryable operation ID, selected
message IDs, and rebuild disposition so startup can finish old-session work
without another host hook. Do not delete it while retaining the corresponding
Atagia conversation.

OpenClaw session JSONL is append-only and may contain abandoned regeneration
branches. Reconciliation follows the parent chain from the current leaf and
sends the full ordered branch, including host message/generation identity, in
one `/selected-transcript` request. Atagia atomically replaces abandoned
suffixes and runs a durable targeted rebuild. The plugin polls that workflow,
performs one automatic authenticated retry when Atagia requests remediation,
and reuses the exact operation ID after ambiguous transport failure. Live
assistant output alone continues to use `/responses`.

Service startup enumerates every pending scope and queries its stable operation
ID. An accepted old-session workflow resumes polling/remediation even after the
host has moved to a new `sessionId`; a confirmed 404 for an unacknowledged
operation replays the temporary outbox with the same ID. Unfinished recovery is
retried on a bounded timer until it reaches a terminal response.

OpenClaw can move its in-memory branch leaf without appending another JSONL
row. The plugin therefore never infers the active leaf from append order. It
uses an explicit leaf ID when the hook supplies one, or uniquely reconstructs
the branch from `before_compaction.messages` or `before_reset.messages`.
Unbranched history may use its sole terminal leaf; ambiguous or missing
active-leaf evidence defers reconciliation without mutating Atagia. Branch
matching uses a linear-time rolling signature and reconstructs only the unique
candidate path, keeping deep legal transcripts within the hook budget.

## Failure And Status Boundaries

Hooks fail open by default. The runtime status contains only operation name,
canonical message ID, source sequence, counts, and a bounded error code. It
does not retain request bodies, prompt/context previews, response text, service
keys, upstream bodies, or raw exception strings. Set `failOpen=false` only when
the deployment intentionally wants hook failures propagated to OpenClaw.

## Contract Verification

```bash
node --test integrations/openclaw/plugin/test.mjs
pytest -q tests/integrations/test_openclaw_plugin.py
```

The Python gate runs OpenClaw's official manifest/loader path, its
`plugins inspect --runtime` command, one complete mock-sidecar turn, service
shutdown, and loader reload. It requires the pinned local OpenClaw checkout.
