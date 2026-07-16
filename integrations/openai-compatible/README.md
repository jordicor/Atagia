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
configured upstream provider, and persists the input, assistant response, and
required root jobs as one durable turn. Transient context failures may continue
without memory; terminal transcript persistence never fails open.

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

The request and response message IDs are an all-or-none pair. Supplying only
one returns `400` before retrieval, storage, or provider execution. A source
sequence is accepted only with its corresponding message ID; the two sequences
remain independently optional.

Supplying neither ID creates a fresh turn on every HTTP request. Supplying both
defines a retryable turn: an identical completed request replays its stored
content, tool calls, finish reason, and provider usage without another context
or provider call, including when replaying between streaming and non-streaming
framing. Reusing the pair with response-determining input changes returns
`409`. A concurrent live owner returns retryable `409 request_in_progress`.
Once a stream has started, an interrupted turn is permanently ambiguous and
the client must retry with a new ID pair.

Tool-only assistant responses and trailing tool-result batches are stored as
ordered, versioned structured metadata. Call IDs, literal argument/result
types, array order, and causal links are part of message identity. Tool data is
treated as untrusted content when projected for retrieval.

Use `live_turn` + `live_prompt_allowed` for normal proxy traffic. Use
`backfill` + `admin_review_only` for offline importers instead of proxy calls.

## Supported Compatibility

- Non-streaming chat completions.
- Streaming SSE chunks, including `stream_options.include_usage`.
- Function/tool calls and `tool_choice="none"`.
- Fail-open handling for transient context infrastructure failures only.
- Durable stable-ID replay and structured tool-call/result transcripts.
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
- A transient context-store outage can continue without memory, while a
  terminal transcript commit failure never emits a successful terminal event.
