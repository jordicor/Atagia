# Open WebUI Integration

Status: implemented, mock-verified, live smoke pending.

`atagia_memory_filter.py` is a copyable Open WebUI Filter Function. It uses
`inlet()` to fetch Atagia context and inject a system message, then uses
`outlet()` to persist the final assistant response when Open WebUI invokes the
outlet hook.

## Install

Open WebUI loads Functions from Python source and auto-detects a top-level
`class Filter`. Import `atagia_memory_filter.py` from Admin Panel -> Functions,
review the code, save it, then attach it globally or to selected models.

Configure valves:

```text
enabled: true
base_url: http://127.0.0.1:8100
api_key: <ATAGIA_SERVICE_API_KEY>
default_user_id: open-webui-user
default_conversation_id: open-webui-default-chat
platform_id: open-webui
user_persona_id:
character_id:
mode: general_qa
memory_privacy_mode: balanced
fail_open: true
emit_debug_status: false
```

## ID Mapping

The filter resolves:

- user from `metadata.atagia_user_id`, `metadata.user_id`, `__user__.id`,
  `__user__.email`, `__user__.name`, then `default_user_id`.
- conversation from `metadata.atagia_conversation_id`,
  `metadata.conversation_id`, `metadata.chat_id`, `body.chat_id`, `body.id`,
  then `default_conversation_id`.
- platform from `platform_id`.
- persona/character from valves.

The filter derives deterministic `message_id`, `source_seq`,
`response_message_id`, and `response_source_seq` values from message order and
content. Edits/regenerations produce new deterministic IDs instead of colliding
with earlier text.

## Debug State

Call `filter.debug_state(__user__, __metadata__, body)` from a host-side debug
console to inspect:

- status,
- resolved IDs,
- last `request_message_id`,
- injected preview,
- fail-open error.

## Important Outlet Caveat

Open WebUI does not run `outlet()` for every API/direct flow. When guaranteed
assistant-response persistence matters for API-direct use, configure the client
to use the Atagia OpenAI-compatible proxy instead of relying on the filter.

## Smoke Checklist

- The filter imports successfully and exposes `Filter`.
- `inlet()` injects an Atagia system block when the sidecar returns context.
- `outlet()` stores an assistant response in normal UI chat flows.
- Atagia down/API error keeps the Open WebUI request working when `fail_open`
  is true.
- Missing or odd chat IDs are encoded safely in URL path segments.
