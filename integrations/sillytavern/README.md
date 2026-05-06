# SillyTavern Integration

Status: native extension scaffold available; proxy path requires explicit host IDs.

SillyTavern is a high-value first native adapter because users already manage
long-running personas, lorebooks, summaries, and vector/RAG extensions. Atagia
fits as an external memory sidecar that can reduce manual memory stacking.

## Best First Path

1. Run Atagia as an HTTP service.
2. Use the native extension so SillyTavern sends stable user, platform,
   character, and chat IDs to Atagia.
3. Use the OpenAI-compatible proxy only with a host setup that can send Atagia
   headers or request metadata.

## Proxy Setup

Start Atagia:

```bash
ATAGIA_SERVICE_MODE=true \
ATAGIA_SERVICE_API_KEY=change-me \
ATAGIA_ADMIN_API_KEY=change-me-admin \
ATAGIA_PROXY_DEFAULT_MODE=companion \
atagia-api --host 127.0.0.1 --port 8100
```

In SillyTavern, configure an OpenAI-compatible source:

```text
API base URL: http://127.0.0.1:8100/v1
API key: change-me
Model: atagia-memory-proxy
Streaming: enabled
```

The proxy requires explicit Atagia identity. A SillyTavern profile must send
`X-Atagia-User-Id`, `X-Atagia-Platform-Id`, and
`X-Atagia-Conversation-Id`, or equivalent `metadata.atagia_*` fields. Use the
native extension path for per-chat memory separation when the profile cannot
set those values.

## Supported Proxy ID Mapping

Atagia resolves IDs in this order:

- User: `X-Atagia-User-Id`, `metadata.atagia_user_id`, request `user`.
- Platform: `X-Atagia-Platform-Id`, `metadata.atagia_platform_id`.
- Conversation: `X-Atagia-Conversation-Id`,
  `metadata.atagia_conversation_id`, `metadata.conversation_id`,
  `metadata.chat_id`.
- Mode: `X-Atagia-Mode`, `metadata.atagia_mode`,
  `ATAGIA_PROXY_DEFAULT_MODE`.

## Native Extension Shape

The scaffold in `extension/`:

- map SillyTavern user/character/chat IDs to stable Atagia IDs,
- calls Atagia before generation through a prompt interceptor,
- inject Atagia context as an internal system block,
- records assistant responses through the `MESSAGE_RECEIVED` event,
- expose enable/disable, base URL, API key, platform, mode, and debug options,
- show an injection/context preview for troubleshooting.

Because it runs in the browser, Atagia needs CORS enabled for the SillyTavern
origin:

```bash
ATAGIA_CORS_ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

## Missing Atagia Pieces

- Live SillyTavern smoke validation against a current install.
- End-user memory review/edit affordances.
- Importer for existing SillyTavern chats/lorebooks/summaries.
