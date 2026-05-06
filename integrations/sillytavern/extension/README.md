# Atagia Memory SillyTavern Extension

Status: scaffold.

This extension uses SillyTavern's prompt-interceptor hook to fetch Atagia
context before generation, inserts an internal memory block into the prompt, and
records generated assistant messages back to Atagia.

## Install

Copy this `extension/` directory into a SillyTavern third-party extension folder
and restart or reload SillyTavern.

For current SillyTavern releases, custom extensions live under either:

```text
data/<user-handle>/extensions/atagia-memory
```

or, for all users:

```text
public/scripts/extensions/third-party/atagia-memory
```

## Atagia Service

Because this is a browser extension, Atagia must allow the SillyTavern origin:

```bash
ATAGIA_SERVICE_MODE=true \
ATAGIA_SERVICE_API_KEY=change-me \
ATAGIA_ADMIN_API_KEY=change-me-admin \
ATAGIA_CORS_ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000 \
atagia-api --host 127.0.0.1 --port 8100
```

Then configure the extension panel:

```text
Atagia base URL: http://127.0.0.1:8100
Service API key: change-me
User ID: sillytavern-user
Conversation prefix: sillytavern
Assistant mode: companion
```

## Limitations

- This scaffold depends on SillyTavern browser extension hooks and should be
  smoke-tested against a live SillyTavern install before public release.
- Response persistence listens for `MESSAGE_RECEIVED`; edited, deleted, and
  swiped messages are not reconciled yet.
- It does not import existing chats, lorebooks, summaries, or vector memories.
