# Aurvek Reference Integration

Aurvek currently has the most complete host-side Atagia integration. The private
Aurvek bridge is a thin fail-open adapter that calls Atagia through the public
`connect_atagia()` facade, then injects Atagia context into Aurvek's existing
LLM flow.

This folder keeps the reusable shape of that integration without depending on
Aurvek's database, admin panel, or template system. It is a copyable example,
not a runtime package. Production code should import the canonical bridge and
helpers from `atagia.integrations`.

Canonical Aurvek conventions now live in `atagia.integrations.aurvek` and are
also exported from `atagia.integrations`:

- `AURVEK_PLATFORM_ID = "aurvek"`
- `aurvek_user_id(id) -> "aurvek:user:{id}"`
- `aurvek_conversation_id(id) -> "aurvek:conv:{id}"`
- `aurvek_message_id(id) -> "aurvek:msg:{id}"`
- `aurvek_prompt_character_id(prompt_id) -> "prompt:{id}"`

## What The Live Aurvek Bridge Does

- Reads host/admin config for enablement, transport, DB path, HTTP URL, service
  API key, admin API key, assistant mode, and timeout.
- Supports `auto`, `local`, and `http` transports.
- Converts Aurvek IDs to namespaced Atagia IDs.
- Calls `create_user()` and `create_conversation()` as a warmup.
- Calls `get_context()` before the host LLM call.
- Passes namespace fields: `platform_id`, `mode`, `character_id`,
  `user_persona_id`, `incognito`, `operational_profile`, and
  `operational_signals`.
- Passes optional stable `message_id` values. Replays with the same role/text
  are idempotent; incompatible reuse returns a conflict from Atagia.
- Passes optional `source_seq` for historical backfills so retrying a failed
  old Aurvek message preserves its order instead of appending it at the end.
- Passes optional `ingest_origin`. Live turns should use the default
  `live_turn`; full-history sync should pass `backfill` so sensitive imported
  candidates go to admin review instead of user confirmation prompts.
- Passes optional `memory_privacy_mode`. Aurvek should default users to
  `balanced`; if a user explicitly chooses `trusted_private`, Atagia treats it
  as broad consent and does not create confirmation/review items solely because
  a memory candidate is sensitive or imported.
- Appends the returned `system_prompt` as an internal memory block.
- Suppresses Aurvek's local long-history window when Atagia context is active.
- Calls `add_response()` after the assistant response is saved.
- Exposes Atagia processing maintenance controls: `pause_new_jobs`,
  `drain_and_pause`, `hard_pause`, and `resume_processing`.
- Atagia may also auto-apply `hard_pause` if recent worker failures dominate
  the configured circuit-breaker window. Aurvek should surface the admin worker
  control state/reason and let an admin resume once provider/network issues are
  fixed.
- Exposes user confirmation helpers and admin review helpers without requiring
  Aurvek to write Atagia internals directly.
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

- Use `source_seq` during full-history sync. Aurvek can usually use its
  conversation-local message order or stable numeric message id if that order is
  monotonic within the conversation.
- Use `ingest_origin="backfill"` for full-history sync/import. Keep the default
  `live_turn` for normal chat paths.
- Wire Settings -> Memory privacy/trust to `memory_privacy_mode`; use
  `balanced` by default and `trusted_private` only for explicit user opt-in.
- Pass first-class attachment metadata to Atagia instead of only text
  placeholders.
- Add an admin-visible context inspector so operators can see what Atagia
  injected.
- Wire the admin stop switch to pause Atagia before DB restores, backups, or
  process restarts.
- Add Settings -> Memory confirmation UX using
  `list_pending_memory_confirmations`, `confirm_pending_memory`, and
  `decline_pending_memory`.
- Add `/admin/atagia` review UX using `list_review_required_memories`,
  `archive_review_required_memory`, and `delete_review_required_memory`.
