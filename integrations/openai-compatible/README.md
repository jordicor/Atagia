# OpenAI-Compatible Memory Proxy

Status: implemented, mock-verified, live smoke pending.

The proxy is the broadest plug-and-play surface for tools that can point at an
OpenAI-compatible base URL but cannot run custom code in the message pipeline.

## Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`
- Server-Sent Events for `stream=true`

The proxy resolves Atagia identity, fetches context for the latest user message,
injects it into the outbound system prompt, forwards the request to the
configured upstream provider, streams or returns the response, and persists the
assistant response fail-open.

## Required Identity

Atagia rejects requests unless it can resolve all three:

- `user_id`
- `platform_id`
- `conversation_id`

Header form:

```text
X-Atagia-User-Id: <stable-host-user-id>
X-Atagia-Platform-Id: <stable-host-platform-id>
X-Atagia-Conversation-Id: <stable-host-chat-id>
X-Atagia-Mode: companion
```

Metadata form for clients that cannot send custom headers:

```json
{
  "metadata": {
    "atagia_user_id": "user_1",
    "atagia_platform_id": "desktop_app",
    "atagia_conversation_id": "chat_1",
    "atagia_mode": "companion"
  }
}
```

## Stable Message Fields

The proxy accepts these fields from either headers or `metadata`:

| Header | Metadata keys |
|---|---|
| `X-Atagia-Message-Id` | `atagia_message_id`, `message_id` |
| `X-Atagia-Source-Seq` | `atagia_source_seq`, `source_seq` |
| `X-Atagia-Response-Message-Id` | `atagia_response_message_id`, `response_message_id` |
| `X-Atagia-Response-Source-Seq` | `atagia_response_source_seq`, `response_source_seq` |
| `X-Atagia-Ingest-Origin` | `atagia_ingest_origin`, `ingest_origin` |
| `X-Atagia-Confirmation-Strategy` | `atagia_confirmation_strategy`, `confirmation_strategy` |
| `X-Atagia-Memory-Privacy-Mode` | `atagia_memory_privacy_mode`, `memory_privacy_mode` |

Use `live_turn` + `live_prompt_allowed` for normal proxy traffic. Use
`backfill` + `admin_review_only` for offline importers instead of proxy calls.

## Supported Compatibility

- Non-streaming chat completions.
- Streaming SSE chunks, including `stream_options.include_usage`.
- Function/tool calls and `tool_choice="none"`.
- Fail-open context retrieval and assistant-response persistence.
- OpenAI-shaped validation, unknown-model, and upstream failure errors.

## Running

```bash
ATAGIA_SERVICE_MODE=true \
ATAGIA_SERVICE_API_KEY=change-me \
ATAGIA_ADMIN_API_KEY=change-me-admin \
ATAGIA_PROXY_MODEL_ID=atagia-memory-proxy \
ATAGIA_LLM_FORCED_GLOBAL_MODEL=anthropic/claude-sonnet-4-6 \
atagia-api --host 127.0.0.1 --port 8100
```

OpenAI-compatible base URL:

```text
http://127.0.0.1:8100/v1
```

Use `change-me` as the OpenAI-compatible API key in the host.

## Smoke Checklist

- `GET /v1/models` returns `atagia-memory-proxy`.
- A non-streaming request with identity fields injects Atagia context.
- A streaming request emits text chunks, a final chunk, optional usage chunk, and
  `data: [DONE]`.
- Tool calls round-trip in both non-streaming and streaming mode.
- Missing `platform_id` is rejected before the upstream model is called.
- Turning Atagia storage off or breaking the sidecar path still returns the model
  response fail-open.
