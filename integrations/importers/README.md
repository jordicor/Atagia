# Atagia Offline Transcript Importers

Status: pre-alpha, contract-tested transcript backfill.

`atagia_importers.py` imports host transcripts through
`POST /v1/conversations/{conversation_id}/messages` with `ingest_origin=backfill`
and `confirmation_strategy=admin_review_only`.

Supported transcript shapes:

- SillyTavern `.jsonl` chat exports with explicit boolean `is_user` fields.
- OpenClaw `messages`, `transcript`, or `sessionFile.messages` arrays whose rows
  contain an explicit `role` of `user` or `assistant`.
- Hermes `messages` or `transcript` arrays with the same explicit roles.

Each call requires stable `host_installation_id`, `host_account_id`, and
`host_conversation_id` values. Message IDs are hashes of that namespace, the
mapped Atagia user, the host message/ordinal, role, and generation. Message text
is never part of identity. Stable host IDs use the shared `host_message`
namespace; old exports without one use a `backfill_message` ordinal. A
SillyTavern live mapping in `extra.atagia_source_identity` is recomputed and
must match, which reconciles the selected exported swipe with its live message
instead of inserting it again.

By default, invalid rows are skipped with visible warnings. A row whose
ingestion request fails counts as `failed`, so for every summary returned by an
importer each input record is accounted for exactly once and
`imported + failed + skipped == total_records`. Pass `strict=True` to validate
the whole batch and raise `ImportBatchValidationError` before the first network
request when roles/content/mappings are invalid; the summary attached to that
exception is a validation report — valid rows were never processed, so they
appear in no counter and the returned-summary invariant does not apply to it.
Malformed JSONL rows and non-object transcript entries retain their batch
position and are reported instead of disappearing from totals.

Hermes `memories` and SillyTavern lorebook entries are curated memory, not chat
turns. They are counted under `curated_memory_unsupported` and never fabricated
as assistant or user messages. A typed curated-memory importer is intentionally
outside this bundle.
