# Hermes Integration

Status: the native provider is contract-tested for straight-line turns and
suffix mutations against Hermes Agent `0.18.2`, commit
`e4ea0a0ed7fc24761b2b425146893561a73216e1`, with Atagia's versioned downstream
host patch. Vanilla Hermes 0.18.2 is explicitly unsupported and rejected.

The native plugin is in `plugins/memory/atagia/`. The required host patch and
its machine-readable pin are in `patches/0.18.2-e4ea0a0/`. See the plugin
[README](plugins/memory/atagia/README.md) for exact apply commands, mandatory
identity settings, mutation ordering, failure behavior, and verification.

The patch adds the `hermes.memory-selection.v1` capability. It uses stable
SessionDB row IDs and supplies the selected transcript/cutoff before context
prefetch, so retry, undo, and regeneration can replace the abandoned Atagia
suffix before any context is read from the new branch. The provider accepts
only a strict retained prefix and never infers mutation state from message text.

`atagia_provider.py` remains a direct Python-library facade for callers that
own their host adapter. It is not the Hermes native plugin and is not part of
the pinned loader contract.

Hermes exports can be backfilled with
`integrations/importers/atagia_importers.py`; only explicit transcript
`user`/`assistant` rows are eligible. Curated `memories` are reported and
skipped rather than fabricated as assistant turns.
