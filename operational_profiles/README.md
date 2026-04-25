# Operational Profiles

Operational Profiles are experimental Phase 0 runtime presets. They describe
device or environment conditions for one request, while `assistant_mode_id`
continues to describe the interaction mode.

Phase 0 is intentionally behavior-neutral: profiles are normalized, authorized,
stored in cache entries, retrieval events, and worker job envelopes, but the
canonical profiles below do not alter retrieval, scoring, extraction, LLM
routing, or worker scheduling.

Canonical preset ids are sealed in Phase 0:

- `normal`
- `low_power`
- `offline`
- `emergency`
- `disaster`

High-risk profiles (`emergency`, `disaster`, or equivalent high-risk effective
signals) require explicit runtime opt-in via settings. Do not add profile JSONs
outside the canonical set until a real integration requires extension.
