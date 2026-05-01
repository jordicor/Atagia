# OpenAI-Compatible Memory Proxy

Status: planned.

This is the most important missing integration surface for broad plug-and-play
usage. Many chat UIs and local model tools can point at an OpenAI-compatible
base URL but cannot run custom Python code in their message pipeline.

## Target Shape

Expose an Atagia-backed endpoint set that looks like an OpenAI-compatible API:

- `GET /v1/models`
- `POST /v1/chat/completions`
- streaming Server-Sent Events for `stream=true`

The proxy would:

1. Receive a normal chat-completion request from the host.
2. Resolve a stable `user_id` and `conversation_id` from headers, request
   metadata, or configured defaults.
3. Fetch Atagia context for the latest user message.
4. Inject the context into the outbound system prompt.
5. Forward the request to the configured upstream provider.
6. Stream the response back to the host.
7. Persist the assistant response in Atagia after completion.

## Why This Matters

This path would make Atagia usable by tools that already support custom API
base URLs, including local chat UIs, roleplay frontends, and model-routing
frontends. It should be implemented in `src/atagia/api/` and
`src/atagia/services/`, not in this examples directory.

## Open Questions

- How should hosts provide durable `user_id` and `conversation_id` values?
- Should the proxy expose one upstream model list or Atagia-specific virtual
  model names?
- How should streaming failures be reconciled with post-response persistence?
- How should multi-user service mode separate tenant API keys from upstream
  provider API keys?
