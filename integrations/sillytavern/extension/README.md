# Atagia Memory SillyTavern Extension

Status: implemented, mock-verified, live smoke pending.

This extension uses SillyTavern's prompt-interceptor hook to fetch Atagia context
before generation, injects that context through `setExtensionPrompt`, and records
generated assistant messages back to Atagia from `MESSAGE_RECEIVED`.

It deliberately does not insert synthetic Atagia messages into `chat`; that would
risk persisting internal memory context as real SillyTavern history.

## Install

Copy this `extension/` directory into a SillyTavern third-party extension folder
and restart or reload SillyTavern.

Current SillyTavern releases commonly use:

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

Configure the extension panel:

```text
Atagia base URL: http://127.0.0.1:8100
Service API key: change-me
User ID: sillytavern-user
Persona ID: optional stable persona ID
Platform ID: sillytavern
Character ID: optional stable character ID
Conversation prefix: sillytavern
Mode: companion
Memory privacy mode: balanced
```

## Debug Panel

The panel shows:

- current status,
- last injected context preview,
- last Atagia request payload,
- last request message ID,
- fail-open error text.

## Smoke Checklist

- `Test connection` succeeds against `/v1/models`.
- A generation calls `/context` with `message_id`, `source_seq`,
  `platform_id`, `character_id`, `ingest_origin`, `confirmation_strategy`, and
  `memory_privacy_mode`.
- `setExtensionPrompt` receives the Atagia context block.
- No Atagia message is added to persistent chat history.
- `MESSAGE_RECEIVED` calls `/responses` with response message ID/source seq.
- Regenerations/swipes produce deterministic response IDs.
- API/CORS errors fail open and appear in the panel.
