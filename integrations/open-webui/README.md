# Open WebUI Integration

Status: filter scaffold available.

Open WebUI is a good target for an Atagia function/filter integration because it
can sit in the request pipeline without replacing the user's model provider.

## Target Shape

- A filter/function that receives the chat payload.
- Stable user and conversation ID mapping from Open WebUI metadata.
- `get_context_for_turn()` before the upstream model call.
- Prompt injection using `build_injection_decision()`.
- `record_assistant_response()` after generation.
- Optional admin-visible debug fields for selected memories and context source.

## Files

- `atagia_memory_filter.py` is a copyable Open WebUI Filter Function.

## Install

Open WebUI loads Functions from Python source and auto-detects a top-level
`class Filter`. Import `atagia_memory_filter.py` from Admin Panel -> Functions,
review the code, save it, then attach it globally or to selected models.

Configure the filter valves:

```text
base_url: http://127.0.0.1:8100
api_key: <ATAGIA_SERVICE_API_KEY>
default_user_id: open-webui-user
default_conversation_id: open-webui-default-chat
platform_id: open-webui
mode: general_qa
```

The filter is toggleable. When enabled for a chat, `inlet()` fetches Atagia
context and injects it into the system prompt. `outlet()` records the final
assistant response back to Atagia.

## Missing Atagia Pieces

- Live Open WebUI smoke validation against a current install.
- Context inspector and memory edit UI.
