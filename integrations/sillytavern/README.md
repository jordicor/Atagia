# SillyTavern Integration

Status: planned.

SillyTavern is a high-value first native adapter because users already manage
long-running personas, lorebooks, summaries, and vector/RAG extensions. Atagia
fits as an external memory sidecar that can reduce manual memory stacking.

## Best First Path

1. Implement the OpenAI-compatible memory proxy.
2. Document the simplest SillyTavern setup that points a connection profile at
   the proxy.
3. Build a native extension only after the proxy path works.

## Native Extension Shape

A native extension should:

- map SillyTavern user/character/chat IDs to stable Atagia IDs,
- call Atagia before generation,
- inject Atagia context as an internal system block,
- avoid duplicating the full local history when Atagia is primary context,
- record assistant responses after generation,
- expose enable/disable, base URL, API key, mode, and debug options,
- show an injection/context preview for troubleshooting.

## Missing Atagia Pieces

- OpenAI-compatible streaming proxy.
- Packaged browser extension scaffold.
- End-user memory review/edit affordances.
- Importer for existing SillyTavern chats/lorebooks/summaries.
