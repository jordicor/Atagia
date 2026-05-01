# Aurvek Reference Integration

Aurvek currently has the most complete host-side Atagia integration. The private
Aurvek bridge is a thin fail-open adapter that calls Atagia through the public
`connect_atagia()` facade, then injects Atagia context into Aurvek's existing
LLM flow.

This folder keeps the reusable shape of that integration without depending on
Aurvek's database, admin panel, or template system.

## What The Live Aurvek Bridge Does

- Reads host/admin config for enablement, transport, DB path, HTTP URL, service
  API key, assistant mode, and timeout.
- Supports `auto`, `local`, and `http` transports.
- Converts Aurvek integer IDs to string Atagia IDs.
- Calls `create_user()` and `create_conversation()` as a warmup.
- Calls `get_context()` before the host LLM call.
- Appends the returned `system_prompt` as an internal memory block.
- Suppresses Aurvek's local long-history window when Atagia context is active.
- Calls `add_response()` after the assistant response is saved.
- Fails open so chat continues if Atagia is disabled or unavailable.

## Files

- `atagia_bridge.py` is a copyable Aurvek-style bridge wrapper backed by
  `atagia.integrations.SidecarBridge`.

## Host-Specific Pieces Not Included Here

Aurvek's production integration also has:

- `atagia_config.py`, which stores settings in Aurvek's `SYSTEM_CONFIG` table.
- `/admin/atagia` routes and a Jinja admin template.
- `ai_calls.py` hooks in the prompt assembly and response persistence pipeline.

Those remain host-specific. The reusable Atagia-side pieces now live in
`src/atagia/integrations/`.

## Next Steps For Aurvek

- Pass first-class attachment metadata to Atagia instead of only text
  placeholders.
- Add an admin-visible context inspector so operators can see what Atagia
  injected.
- Add a memory review/edit path once Atagia exposes stable edit/reclassify APIs.
