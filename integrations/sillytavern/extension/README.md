# Atagia Memory Browser Extension

Compatibility target: SillyTavern `1.18.0` at commit
`51ad27fb86d39a3daca3adaa970375c9670c12df`.

This browser extension requires the companion Atagia server plugin in
`../server-plugin/`. It has no service-key, Atagia-base-URL, platform-ID, or
Atagia-user-ID setting. It calls only the fixed same-origin server route and
uses the SillyTavern session and CSRF token.

This is a reference integration: it is not yet production-supported until a real
user/assistant turn has been verified against the target SillyTavern install.
Local automated verification is contract-level only, not a live-host smoke.

## Install

First install and configure the server plugin as described in
`../server-plugin/README.md`. Then copy this directory to a supported
SillyTavern extension location, such as:

```text
<SillyTavern>/data/<user-handle>/extensions/atagia-memory
```

Restart or reload SillyTavern. The extension requires SillyTavern `1.18.0` and
does not support automatic client or server-plugin drift from the pinned host.

## Settings

The panel exposes only non-secret behavior settings:

- enable/disable;
- optional persona and character IDs;
- conversation prefix;
- mode;
- memory privacy mode; and
- memory-only diagnostic metadata opt-in.

The Atagia user is selected server-side from the authenticated SillyTavern user
handle. Browser payload fields cannot override that mapping.

When upgrading from the legacy extension, loading settings allowlists the
current non-secret fields and deletes everything else immediately, including
persisted `apiKey`, `lastRequest`, and `lastPreview` values. It also removes the
old base URL, user/platform identity, request-message ID, status, and error
fields. A service key that was ever persisted in the browser should not be
reused; provision a fresh server-side key — see `../PRODUCTION_CREDENTIALS.md`.

## Runtime Behavior

Before generation, the extension sends the current user message to the fixed
same-origin `/context` route and injects returned context with
`setExtensionPrompt`. It does not insert synthetic messages into persistent chat
history. After the host's response event, it sends the assistant response to
the fixed same-origin `/response` route. Atagia or boundary failures fail open
for chat generation and expose only a generic status code. The server-confirmed
mapping is saved in the host message: duplicate events and reloads do not resend
that generation, while a failed request remains retryable.

Debug mode retains at most 20 metadata-only events in module memory. It does
not retain or display message text, assistant text, request payloads, system
prompts, memory previews, credentials, user mappings, or upstream response
bodies. Disabling debug, reloading/page hiding, logging out, or replacing the
module clears this memory and any injected extension prompt.

## Validation Checklist

- **Test server boundary** succeeds without an Atagia credential in browser
  requests or storage.
- Legacy settings are purged on upgrade and saved without secret/content
  fields.
- Another browser extension cannot read an Atagia key or mapped Atagia user ID
  from extension settings, DOM, storage, module runtime, or network headers.
- Reload and logout clear diagnostics and the injected prompt.
- If a key was previously exposed to the browser: a server-side probe using
  that old key returns `401`, while the same-origin route works with the fresh
  server-held key before and after restarting both Atagia and SillyTavern.
