# Atagia Offline Importers

Status: importer-ready and mock-verified.

`atagia_importers.py` is a copyable, standard-library-only importer module for
platform exports. Every importer writes text messages through:

```text
POST /v1/conversations/{conversation_id}/messages
```

with:

```json
{
  "ingest_origin": "backfill",
  "confirmation_strategy": "admin_review_only"
}
```

Supported sources:

- SillyTavern `.jsonl` chat exports.
- SillyTavern lorebook text pasted/exported as plain text.
- OpenClaw session transcripts shaped as `messages`, `transcript`, or
  `sessionFile.messages`.
- Hermes session transcripts and text memory exports when shaped as
  `messages`, `transcript`, or `memories`.

Importers are rerunnable: message IDs and source sequences are deterministic for
the same input text, role, and source order.
