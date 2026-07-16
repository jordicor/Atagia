# OpenClaw Integration

Status: native OpenClaw `2026.5.6` contract verified; real deployed sidecar
smoke pending.

OpenClaw remains the source of truth for sessions, transcript branches, tools,
and host lifecycle. Atagia retrieves advisory continuity context and stores
host conversation turns.

## Files

- `plugin/` is the native JavaScript package with the required manifest,
  package discovery metadata, typed hooks, and durable identity reconciliation.
- `atagia_adapter.py` remains a Python facade for Python-hosted experiments; it
  is not the OpenClaw loader gate.

## Native Lifecycle

The JavaScript plugin uses the real host surfaces:

- `before_prompt_build` → Atagia `/context` → `{ prependContext }`;
- `agent_end` → final assistant message → Atagia `/responses`;
- `before_compaction`, `before_reset`, and `session_end` → selected-branch transcript
  replacement through Atagia `/selected-transcript`, followed by bounded status
  polling and durable remediation retry.

The manifest and runtime are pinned to OpenClaw `2026.5.6`, commit
`8934095c828de8d6268e0e42d8cfe6651ccf5a1b`. Unsupported host upgrades require
an explicit contract review and version update.

The plugin requires stable installation/account identity and an explicit
OpenClaw-account-to-Atagia-user mapping. Canonical message IDs contain those
scopes plus immutable `sessionId`, host event/message, role, and generation;
they never contain message text. Reusable `sessionKey` values remain routing
metadata only. Durable source mappings reconcile live `runId` events with the
selected transcript after restart and keep ordering monotonic.

Append order is not branch selection: OpenClaw can navigate to an older leaf
without writing another row. The plugin uses active hook messages or an
explicit leaf ID to resolve the selected path. It may infer the sole terminal
leaf for a linear transcript, but defers when a branched `session_end` provides
only append-only history. The native `before_reset` hook supplies the active
selection before `/new` or `/reset` discards that branch.
Selection epochs, sanitized scope coordinates, retryable operation IDs, and a
temporary pre-acknowledgement outbox live in the durable plugin store. Startup
resumes old-session workflows, so a reset/restart cannot strand Atagia's
availability fence or turn an ambiguous request into a different branch
mutation.

See `plugin/README.md` for exact configuration, identity-store semantics, hook
policy, and verification commands.

## Offline Importer

`integrations/importers/atagia_importers.py` imports OpenClaw transcript exports
only when every accepted row has an explicit `user` or `assistant` role. It
uses the same canonical external identity schema. When a host export cannot
carry the live mapping sidecar, run the plugin's selected-transcript
reconciliation before a separate offline import to avoid treating the two
surfaces as interchangeable.

## Remaining Deployment Gate

The local official loader, runtime inspection, complete hook lifecycle, and
shutdown/reload pass with a bounded mock Atagia HTTP server. This is not a claim
that an OpenClaw Gateway and a real Atagia service have completed a live model
turn. Before treating a deployment as verified, run one live Gateway turn
against the real Atagia service, verify context injection and response
persistence, restart both services, and repeat the turn. That operational gate
is still pending.
