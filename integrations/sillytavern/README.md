# SillyTavern Integration

Compatibility target: SillyTavern `1.18.0` at commit
`51ad27fb86d39a3daca3adaa970375c9670c12df`.

The same-origin credential/retention boundary and the pinned assistant event
contract are implemented and contract-tested locally. This is a reference
integration: it is not yet production-supported, because a full user/assistant
turn has not been run against a live SillyTavern install (restart-safe). Verify
that turn on the target installation before relying on it in production.

The Atagia service key is held server-side only. Provision a dedicated key for
the deployment; if a key was ever exposed to a browser or logs, do not reuse it
— see `PRODUCTION_CREDENTIALS.md`.

## Components

- `server-plugin/` is a SillyTavern server plugin. It owns the Atagia service
  key, maps the authenticated SillyTavern user to an Atagia user, and exposes
  fixed same-origin routes to the browser extension.
- `extension/` is the browser extension. It may persist only non-secret
  behavior settings and never accepts an Atagia service key or Atagia user ID.
- `PRODUCTION_CREDENTIALS.md` describes provisioning the single server-side
  service key, and replacing a key that was ever exposed to a browser or logs.

The server plugin must be installed and configured before enabling the browser
extension. The browser calls only `/api/plugins/atagia-memory/*`; it does not
call Atagia directly, so this integration does not require an Atagia CORS
allowlist for the SillyTavern origin.

## Install

Use the pinned SillyTavern version and disable automatic server-plugin updates:

```yaml
enableServerPlugins: true
enableServerPluginsAutoUpdate: false
disableCsrfProtection: false
```

Copy `server-plugin/` to:

```text
<SillyTavern>/plugins/atagia-memory
```

Copy `extension/` to one of SillyTavern's supported extension locations, for
example the current user's directory:

```text
<SillyTavern>/data/<user-handle>/extensions/atagia-memory
```

Configure these values only in the server process or its secret manager:

```bash
ATAGIA_BASE_URL=https://atagia.internal.example
ATAGIA_SERVICE_API_KEY=<replacement-key-from-secret-manager>
ATAGIA_SILLYTAVERN_INSTALLATION_ID=<stable-unique-deployment-id>
ATAGIA_SILLYTAVERN_USER_MAP='{"alice":"atagia-user-alice","bob":"atagia-user-bob"}'
ATAGIA_SILLYTAVERN_TIMEOUT_MS=20000
```

The installation ID is a stable, non-secret identifier unique to this
SillyTavern deployment. The user-map keys are exact authenticated SillyTavern
handles; its values are server-assigned Atagia user IDs. Do not put the mapping
or key in the browser extension, extension settings, HTML, or client-side
JavaScript.

## Message Identity And Regeneration

The browser stores a random host message ID in each SillyTavern message's
`extra` metadata before contacting the boundary. The server then constructs the
final Atagia ID from integration kind, installation ID, authenticated handle,
mapped Atagia user, raw host chat ID, host message ID, role, and selected
generation. Message text is never an identity input. `source_seq` is the
one-based chronological chat position for ordinary turns and is omitted when a
swipe/continuation reuses a position.

The pinned `MESSAGE_RECEIVED` contract emits primitive `(messageId, type)`
arguments. The extension resolves `getContext().chat[messageId]`; it never
searches by response text. Behavior is:

| Host action | Atagia identity |
| --- | --- |
| Normal generation or retry | Same host message/generation; retry is idempotent. |
| Later message with repeated text | New persisted host message ID; distinct turn. |
| Regenerate/new swipe | Same host message ID plus distinct swipe/generation; selected mapping changes. |
| Continue | Same host message plus host generation marker; synthetic sequence is omitted. |
| JSONL backfill with live metadata | Recomputes the selected live ID and reuses its stored sequence/null sequence. |
| Old JSONL without live metadata | Uses the explicit `backfill_message` ordinal namespace. |

Every mapping is saved in the chat before the network request and the
server-confirmed Atagia ID is saved afterward. Reloads therefore reuse the same
identity. A repeated event for an already confirmed generation performs no
second response request; an event whose prior request failed remains retryable.
Chat changes clear the injected prompt before the next turn.

Restart SillyTavern after installing or changing the server plugin. In the
Atagia Memory panel, use **Test server boundary**, then enable the integration.

## Security Boundary

The server plugin requires an authenticated SillyTavern session, the matching
session user, same-origin browser context when supplied by the browser, and the
SillyTavern CSRF token. It does not accept browser authority: authority fields
in JSON are rejected, and Atagia identity or authorization headers supplied by
the browser are never forwarded. The server constructs Atagia authorization
and identity headers from its own configuration.

The browser extension stores only:

- enabled state;
- persona and character display IDs;
- conversation prefix;
- Atagia mode and memory privacy mode; and
- the debug opt-in flag.

An upgrade allowlists the current non-secret settings and deletes everything
else, including legacy `apiKey`, `lastRequest`, and `lastPreview` fields and
superseded endpoint, identity, and status fields. Debug diagnostics contain
bounded metadata only, remain in memory, and are cleared when debug is disabled,
on reload/page hide, and on logout. Message text, assistant responses, system
prompts, memory previews, request bodies, and upstream error bodies are not
retained as diagnostics or rendered in the settings panel.

## Operations

The Atagia service key is held only by the server plugin. Provision a dedicated
key for the deployment. If an earlier browser extension build ever persisted a
service key, do not reuse that key — provision a new one and confirm Atagia no
longer accepts the old value. Deleting the browser setting alone does not
account for copies made elsewhere. See
[PRODUCTION_CREDENTIALS.md](PRODUCTION_CREDENTIALS.md).

Atagia accepts one service key for this boundary; there is no dual-key window.

JSONL transcript backfill is available through
`integrations/importers/atagia_importers.py`. Lorebooks are curated memory, so
the importer reports and skips them rather than fabricating chat turns.
