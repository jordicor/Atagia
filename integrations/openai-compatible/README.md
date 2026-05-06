# OpenAI-Compatible Memory Proxy

Status: MVP implemented.

This is the most important missing integration surface for broad plug-and-play
usage. Many chat UIs and local model tools can point at an OpenAI-compatible
base URL but cannot run custom Python code in their message pipeline.

## Target Shape

Atagia now exposes an Atagia-backed endpoint set that looks like an
OpenAI-compatible API:

- `GET /v1/models`
- `POST /v1/chat/completions`
- streaming Server-Sent Events for `stream=true`

The proxy:

1. Receive a normal chat-completion request from the host.
2. Resolve stable `user_id`, `platform_id`, and `conversation_id` values from
   headers or request metadata.
3. Fetch Atagia context for the latest user message.
4. Inject the context into the outbound system prompt.
5. Forward the request to the configured upstream provider.
6. Stream the response back to the host when `stream=true`.
7. Persist the assistant response in Atagia after completion.

## Running The Proxy

Start Atagia as an HTTP service:

```bash
atagia-api --host 127.0.0.1 --port 8100
```

Required service-mode environment:

```bash
ATAGIA_SERVICE_MODE=true
ATAGIA_SERVICE_API_KEY=change-me
ATAGIA_ADMIN_API_KEY=change-me-admin
ATAGIA_PROXY_MODEL_ID=atagia-memory-proxy
ATAGIA_LLM_FORCED_GLOBAL_MODEL=anthropic/claude-sonnet-4-6
```

The visible OpenAI-compatible base URL is:

```text
http://127.0.0.1:8100/v1
```

Use the Atagia service API key as the OpenAI-compatible API key. Hosts that can
send custom headers should send:

```text
X-Atagia-User-Id: <stable-host-user-id>
X-Atagia-Platform-Id: <stable-host-platform-id>
X-Atagia-Conversation-Id: <stable-host-chat-id>
X-Atagia-Mode: companion
```

Hosts that cannot send custom headers can use request `metadata` keys instead:

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

Tools that cannot send headers or metadata cannot safely use the proxy directly
because Atagia will not synthesize a conversation or platform identity.

## Why This Matters

This path makes Atagia usable by tools that already support custom API
base URLs, including local chat UIs, roleplay frontends, and model-routing
frontends. The code lives in `src/atagia/api/routes_openai_proxy.py` and
`src/atagia/services/openai_proxy_service.py`.

## Open Questions

- Should the proxy expose multiple virtual model names or keep one configured
  model id?
- How should streaming failures be reconciled with post-response persistence?
- How should multi-user service mode separate tenant API keys from upstream
  provider API keys?
- How much OpenAI tool-calling compatibility should be implemented before a
  real host needs it?
