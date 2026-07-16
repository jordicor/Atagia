# Hermes 0.18.2 Memory Selection Patch

This directory is pinned to Hermes Agent commit
`e4ea0a0ed7fc24761b2b425146893561a73216e1` and adds the capability contract
`hermes.memory-selection.v1` required by Atagia's native memory provider.

Apply only from that exact clean commit:

```bash
test "$(git rev-parse HEAD)" = "e4ea0a0ed7fc24761b2b425146893561a73216e1"
git apply --check hermes-memory-selection-v1.patch
git apply hermes-memory-selection-v1.patch
```

`PATCH_METADATA.json` is the machine-readable version/commit/capability pin.
`memory_selection.py` mirrors the new host module embedded in the patch and is
compared byte-for-byte by the repository contract test to prevent drift.
The patch is intentionally downstream: it does not claim that upstream vanilla
Hermes exposes this contract.
