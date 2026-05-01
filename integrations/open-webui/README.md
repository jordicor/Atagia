# Open WebUI Integration

Status: planned.

Open WebUI is a good target for an Atagia function/filter integration because it
can sit in the request pipeline without replacing the user's model provider.

## Target Shape

- A filter/function that receives the chat payload.
- Stable user and conversation ID mapping from Open WebUI metadata.
- `get_context_for_turn()` before the upstream model call.
- Prompt injection using `build_injection_decision()`.
- `record_assistant_response()` after generation.
- Optional admin-visible debug fields for selected memories and context source.

## Missing Atagia Pieces

- A concrete packaged Open WebUI function/filter file.
- Operator docs for service mode, API keys, and local mode.
- Context inspector and memory edit UI.
