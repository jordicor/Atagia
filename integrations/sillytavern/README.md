# SillyTavern Integration

Status: implemented, mock-verified, live smoke pending.

SillyTavern is a high-value native adapter because users already manage
long-running characters, personas, lorebooks, summaries, and RAG extensions.
Atagia fits as an external memory sidecar that retrieves continuity context and
stores long-horizon conversation memory.

## Recommended Path

1. Run Atagia as an HTTP service.
2. Install the native extension in `extension/`.
3. Configure stable user, platform, persona, character, and chat IDs.
4. Use the OpenAI-compatible proxy only when a SillyTavern/OpenAI-compatible
   setup can send Atagia headers or request `metadata`.

## Native Extension

The extension:

- persists an Atagia conversation ID in `chatMetadata.atagia_conversation_id`,
- maps SillyTavern chat/persona/character identity into Atagia identity,
- calls `/v1/conversations/{id}/context` before generation,
- injects memory through `setExtensionPrompt` rather than mutating persisted chat
  history,
- records assistant responses from `MESSAGE_RECEIVED`,
- derives deterministic IDs/source sequences that are rerunnable and tolerate
  edits/regenerations,
- exposes last request, resolved message ID, injected preview, status, and
  fail-open errors in the settings panel.

Because it runs in the browser, Atagia needs CORS enabled for the SillyTavern
origin:

```bash
ATAGIA_CORS_ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

## Proxy Setup

Start Atagia:

```bash
ATAGIA_SERVICE_MODE=true \
ATAGIA_SERVICE_API_KEY=change-me \
ATAGIA_ADMIN_API_KEY=change-me-admin \
ATAGIA_PROXY_DEFAULT_MODE=companion \
atagia-api --host 127.0.0.1 --port 8100
```

OpenAI-compatible settings:

```text
API base URL: http://127.0.0.1:8100/v1
API key: change-me
Model: atagia-memory-proxy
Streaming: enabled
```

The proxy still requires explicit Atagia identity:

- `X-Atagia-User-Id`
- `X-Atagia-Platform-Id`
- `X-Atagia-Conversation-Id`

or equivalent `metadata.atagia_*` fields.

## Importers

Use `integrations/importers/atagia_importers.py` for SillyTavern `.jsonl` chat
exports and plain text lorebook imports. These backfills are admin-review-only
and rerunnable.

## Smoke Checklist

- Extension loads and settings persist.
- First generation calls Atagia and uses `setExtensionPrompt`.
- Chat history length does not change from Atagia context injection.
- Assistant responses persist once per generated message/swipe.
- Atagia down/API error leaves generation working with visible fail-open status.
- JSONL/lorebook importer can replay the same file without duplicate messages.
