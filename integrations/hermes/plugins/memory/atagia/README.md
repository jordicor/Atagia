# Atagia Hermes Memory Provider

Status: contract-tested against Hermes Agent `0.18.2` at commit
`e4ea0a0ed7fc24761b2b425146893561a73216e1`, with the required downstream
`hermes.memory-selection.v1` patch. Unpatched/vanilla Hermes is rejected before
the provider can read context or write a message.

## Exact Host Requirement

The `MemoryProvider` ABI in Hermes 0.18.2 accepts lifecycle keyword arguments,
but the vanilla host does not expose its selected transcript before `prefetch`
and does not provide a trustworthy cutoff for `/retry` or `/undo`. Install the
versioned patch shipped in this repository:

```bash
cd /path/to/hermes-agent
test "$(git rev-parse HEAD)" = "e4ea0a0ed7fc24761b2b425146893561a73216e1"
git apply --check /path/to/atagia/integrations/hermes/patches/0.18.2-e4ea0a0/hermes-memory-selection-v1.patch
git apply /path/to/atagia/integrations/hermes/patches/0.18.2-e4ea0a0/hermes-memory-selection-v1.patch
```

Do not apply this patch to another Hermes revision. The provider validates the
advertised Hermes version, commit, and capability contract during
`initialize()`. A host that omits or changes any of those values raises
`UnsupportedHermesHostError`, reports
`hermes_memory_selection_capability_missing`, and remains unavailable.

The patch makes five bounded host changes:

- advertises `hermes.memory-selection.v1` during provider initialization;
- derives message identity from the durable, autoincrementing SessionDB row ID;
- passes the active selected transcript and retained cutoff before every
  `prefetch`;
- soft-deletes the abandoned SessionDB suffix for `/retry`, and signals retry
  before its replacement turn;
- sends authoritative undo state immediately and supplies the final selected
  transcript at `sync_turn`.

The same contract accepts `regeneration` as an explicit mutation kind for host
surfaces that regenerate without using the CLI `/retry` command.

## Install And Configure

Copy this `atagia/` directory to the patched Hermes installation's
`plugins/memory/` directory or to `$HERMES_HOME/plugins/`, then select it:

```yaml
memory:
  provider: atagia
```

Set all identity values before Hermes discovers providers:

```bash
ATAGIA_BASE_URL=http://127.0.0.1:8100
ATAGIA_SERVICE_API_KEY=<server-side-service-key>
ATAGIA_HERMES_INSTALLATION_ID=<stable-unique-Hermes-deployment-id>
ATAGIA_HERMES_HOST_ACCOUNT_ID=<stable-Hermes-profile-or-account-id>
ATAGIA_HERMES_USER_ID=<mapped-Atagia-user-id>
```

All values are mandatory. `user_id_alt` or `user_id` supplied by a Hermes
gateway changes the host-account scope for that runtime; it does not replace
the mapped Atagia user. The service key remains server-side.

## Mutation And Prefetch Ordering

For a normal turn, the host persists the current user row first and sends:

1. the complete selected user/assistant prefix;
2. its last durable host row ID as the retained cutoff;
3. the current durable user row and mutation kind.

The provider validates that signal before opening a turn identity. For retry,
undo, or regeneration it first calls Atagia's
`atagia.selected-transcript.v1` replacement endpoint. If rebuilding is still in
progress, it polls the exact operation before requesting context. It never
calls `/context` against the abandoned branch. A timeout or transient status
failure returns no memory context; a malformed signal, failed replacement, or
server `remediation_required` state durably disables the mapped identity scope.
Hermes generation itself remains fail-open.

An empty terminal assistant row or a user row left without a terminal assistant
is forwarded as an invalid boundary and durably rejected. It is never shortened
into an apparently complete selected prefix and is never backfilled during
session reconciliation.

Only suffix mutations are supported: the new selection must be a strict prefix
of the previously selected host row IDs. Arbitrary branch splicing is rejected.
Start the integration on a fresh Hermes session and retain
`$HERMES_HOME/atagia/identity.sqlite3`; deleting it while the mapped Atagia
conversation exists destroys the provider's selection epoch and identity
mapping. First attachment in the middle of an active session is rejected.

## Identity And Delivery

Atagia message IDs hash the integration, installation/account, mapped user,
Hermes session, durable host row ID, role, and generation. Text is never an
identity input. A later repeated message is distinct, while replaying the same
full host tuple is idempotent. `source_seq` is a monotonic conversation ordinal;
after a rewind it can contain gaps but is never reused.

The local SQLite database stores identity mappings, selected host row IDs,
selection epochs, and delivery acknowledgements, but not transcript text or
retrieved context. A mapping is confirmed only when Atagia returns the exact
message ID and sequence. Uncertain delivery is retried with the same identity.

`prefetch` is synchronous because Hermes needs its returned context before
generation. `sync_turn` writes through one ordered background worker. Shutdown
stops new work and gives the worker a bounded drain window; if work remains,
the worker retains SQLite ownership until it exits. Curated Hermes memories are
not transcript rows and remain an intentional no-op.

## Verification

Repository tests use the faithful 0.18.2 `MemoryProvider` snapshot, its strict
`register(ctx)` loader, a patched lifecycle driver with SessionDB-style row IDs,
and a reverse-apply check against a clean checkout of the exact pinned commit
when that checkout is available. They cover straight-line turns, repeated text,
restart, retry, undo, regeneration, rebuild polling, malformed/missing signals,
delivery retry, and bounded shutdown.
