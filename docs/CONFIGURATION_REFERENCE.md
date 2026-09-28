# Configuration Reference

Full reference for every `ATAGIA_*` environment variable read by the Atagia
runtime. This is the canonical companion to the project root
[`README.md`](../README.md), which links here for anything beyond the minimum
quickstart configuration.

Atagia loads environment variables from the process environment first and
then from a `.env` file at the repository root (process values win). All
configuration is centralized in
[`src/atagia/core/config.py`](../src/atagia/core/config.py), with a few
runtime-only variables read by the MCP server, the client SDK, and the
sidecar bridge.

The fastest way to bootstrap a working `.env` is:

```bash
cp .env.example .env
# configure the provider keys required by your selected model routing
# set distinct ATAGIA_SERVICE_API_KEY and ATAGIA_ADMIN_API_KEY values
# because the shipped example enables service mode
```

---

## 1. Required for default runtime

The default model selection routes ingest and compaction components to direct
MiniMax M3, retrieval components to OpenRouter (stable Gemini Flash-Lite),
ordinary chat answers to OpenRouter (DeepSeek v4 Flash), and
privacy/consent/export-sensitive components to Anthropic (Claude Sonnet 4.6).
Benchmark CLIs also default to direct Kimi K2.7 Code for judging so evaluator
schema/JSON instability does not hide product or retrieval failures.
The minimum viable configuration therefore needs:

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_MINIMAX_API_KEY` | _(unset)_ | Required for default ingest/compaction components | API key or subscription key for direct MiniMax API access (used by default ingest and compaction intelligence such as `extractor`, `text_chunker`, and `compactor`). |
| `ATAGIA_OPENROUTER_API_KEY` | _(unset)_ | Required for default retrieval/chat components | API key for OpenRouter (used by default retrieval components and by the default `chat` component). |
| `ATAGIA_ANTHROPIC_API_KEY` | _(unset)_ | Required for default privacy/consent/export components | API key for Anthropic Claude (used by `summary_privacy_judge`, `summary_privacy_refiner`, `consent_confirmation`, and `export_anonymizer`). |
| `ATAGIA_KIMI_API_KEY` | _(unset)_ | Required for benchmark CLIs with their default judge | API key for direct Kimi API access (used by the default benchmark judge `kimi/kimi-k2.7-code`). |

To run all completion components on a single provider instead, set
`ATAGIA_LLM_FORCED_GLOBAL_MODEL` (see [LLM model selection](#4-llm-model-selection)) and
provide only that provider's API key.

For quality and cost balance, prefer role-specific routing over a single
forced-global model. The defaults route ingest and compaction intelligence to
`minimax/MiniMax-M3`, retrieval intelligence to
`openrouter/google/gemini-3.1-flash-lite`, cheap ordinary-answer generation to
`openrouter/deepseek/deepseek-v4-flash`, and privacy/consent/export-sensitive
components to `anthropic/claude-sonnet-4-6`. For local experiments, point an
explicit local endpoint catalog at an OpenAI-compatible server and use a model
spec such as `local/desktop/qwen3-coder:30b`.
Use the stable `openrouter/google/gemini-3.1-flash-lite` slug, not the
deprecated `-preview` endpoint, for retrieval overrides.

---

## 2. Storage

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_SQLITE_PATH` | `./data/atagia.db` | Optional | Path to the SQLite database used as the single source of truth. Applies to library mode, service mode, the MCP server, and the client SDK alike. |
| `ATAGIA_DB_PATH` | `ATAGIA_SQLITE_PATH` | Optional | SQLite path override for the MCP server and the client SDK. Unset, both fall through to `ATAGIA_SQLITE_PATH` and its default. |
| `ATAGIA_STORAGE_BACKEND` | `inprocess` | Optional | Storage backend selector. `inprocess` keeps streams in memory; `redis` uses Redis Streams. |
| `ATAGIA_REDIS_URL` | `redis://localhost:6379/0` | Optional | Redis connection URL when `ATAGIA_STORAGE_BACKEND=redis`. |
| `ATAGIA_MIGRATIONS_PATH` | Packaged resources | Optional | Explicit custom directory containing numbered SQL migration files. |
| `ATAGIA_MANIFESTS_PATH` | Packaged resources | Optional | Explicit custom directory containing assistant mode manifest JSON files. |
| `ATAGIA_OPERATIONAL_PROFILES_PATH` | Packaged resources | Optional | Explicit custom directory containing operational profile JSON files. |
| `ATAGIA_ARTIFACT_BLOB_STORAGE_KIND` | `sqlite_blob` | Optional | Artifact blob backend. `sqlite_blob` is the only supported runtime value. |
| `ATAGIA_ARTIFACT_BLOB_STORAGE_PATH` | `./data/artifact_blobs` | Optional | Legacy blob root used only by the offline local-file migration command. |

Deployments that previously used the retired local-file backend must stop all
Atagia writers and lifecycle/GC workers, then run:

```bash
atagia-artifact-blob-migrate \
  --sqlite-path /path/to/atagia.db \
  --artifact-blob-storage-path /path/to/legacy/artifact_blobs \
  run
```

The command inventories every declared blob-reference table, verifies file
hashes and sizes, migrates references to SQLite in resumable row commits, and
unlinks a file only after a committed global zero-reference check. Normal
startup refuses legacy references, pending file deletions, or undrained cleanup
intents and reports the migration command to run.

### Durable worker dispatch and execution leases

SQLite stores every complete validated recovery envelope and is authoritative
for queued, claimed, running, retrying, deferred, and terminal job state. The
selected storage backend carries only opaque wake-up notifications. A startup
sweep and the continuous dispatcher recover nonterminal rows; Redis reset or
notification loss therefore does not lose canonical work.

Each worker renews an execution lease while it owns a job. Terminal writes,
retries, and worker-created child jobs compare the owner, fence token,
lifecycle epoch, and an unexpired lease. A stale owner may incur duplicate
provider cost across a crash boundary, but cannot commit the logical effect.
`inprocess` is a supported single-process backend only; configure Redis when
more than one service or worker process participates.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_WORKERS_ENABLED` | `false` | Optional | Start the durable dispatcher and background workers in this process. Jobs are still committed durably when workers are disabled. |
| `ATAGIA_SERVICE_PROCESS_COUNT` | `1` (or `WEB_CONCURRENCY`) | Optional | Declared service-process count. Values above one require the Redis storage backend. |
| `ATAGIA_WORKER_DISPATCH_VISIBILITY_SECONDS` | `30` | Optional | Lifetime of one SQLite dispatch claim before another dispatcher may recover it. |
| `ATAGIA_WORKER_DISPATCH_SWEEP_INTERVAL_SECONDS` | `0.5` | Optional | Maximum interval between continuous scans for publishable durable jobs. |
| `ATAGIA_WORKER_DISPATCH_BATCH_SIZE` | `100` | Optional | Maximum durable rows considered in one dispatcher sweep. |
| `ATAGIA_WORKER_EXECUTION_LEASE_SECONDS` | `120` | Optional | Renewable SQLite execution-lease lifetime. |
| `ATAGIA_WORKER_EXECUTION_HEARTBEAT_SECONDS` | `30` | Optional | Lease-renewal cadence; it must be shorter than the execution lease. |
| `ATAGIA_WORKER_STREAM_RECLAIM_IDLE_SECONDS` | `120` | Optional | Minimum idle time before reclaiming an abandoned transient delivery; it must not be shorter than dispatch visibility. |
| `ATAGIA_WORKER_RETRY_BACKOFF_INITIAL_SECONDS` | `1` | Optional | Initial durable retry delay after a retryable worker failure. |
| `ATAGIA_WORKER_RETRY_BACKOFF_MAX_SECONDS` | `30` | Optional | Maximum durable retry delay. |
| `ATAGIA_WORKER_TRANSIENT_DEFER_SECONDS` | `60` | Optional | Initial delay for transient deferral when the worker circuit is unavailable. |
| `ATAGIA_WORKER_TRANSIENT_DEFER_MAX_SECONDS` | `300` | Optional | Maximum per-deferral delay. |
| `ATAGIA_WORKER_TRANSIENT_DEFER_MAX_COUNT` | `12` | Optional | Maximum transient deferrals before the job follows its bounded terminal policy. |
| `ATAGIA_WORKER_TRANSIENT_DEFER_MAX_AGE_SECONDS` | `3600` | Optional | Maximum wall-clock age allowed for transient deferral. |

---

## 3. LLM provider keys and base URLs

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_ANTHROPIC_API_KEY` | _(unset)_ | Conditional | API key for Anthropic. Required when any resolved component uses an `anthropic/...` model. |
| `ATAGIA_OPENAI_API_KEY` | _(unset)_ | Conditional | API key for OpenAI. Required when any component uses an `openai/...` model or for OpenAI-hosted embeddings. |
| `ATAGIA_GOOGLE_API_KEY` | _(unset)_ | Conditional | API key for Google Gemini. Required when any component uses a `google/...` model. |
| `ATAGIA_KIMI_API_KEY` | _(unset)_ | Conditional | API key for direct Kimi/Moonshot API access. Required when any component or benchmark judge uses a `kimi/...` model such as `kimi/kimi-k2.7-code`. |
| `ATAGIA_MINIMAX_API_KEY` | _(unset)_ | Conditional | API key or subscription key for direct MiniMax API access. Required when any component uses a `minimax/...` model such as `minimax/MiniMax-M3`. |
| `ATAGIA_OPENROUTER_API_KEY` | _(unset)_ | Conditional | API key for OpenRouter. Required when any component uses an `openrouter/...` model. |
| `ATAGIA_TYPESAFE_API_KEY` | _(unset)_ | Conditional | TypeSafe API key for native finite-choice decisions. Required when an eligible component uses `typesafe/jev-latest`. |
| `ATAGIA_ANTHROPIC_BASE_URL` | _(unset)_ | Optional | Override Anthropic API base URL. |
| `ATAGIA_OPENAI_BASE_URL` | _(unset)_ | Optional | Override OpenAI API base URL. |
| `ATAGIA_OPENAI_EMBEDDING_BASE_URL` | _(unset)_ | Optional | Override only the OpenAI-compatible embeddings base URL. When unset, embeddings use `ATAGIA_OPENAI_BASE_URL`. |
| `ATAGIA_KIMI_BASE_URL` | `https://api.moonshot.ai/v1` | Optional | Override the direct Kimi/Moonshot OpenAI-compatible API base URL. |
| `ATAGIA_MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | Optional | Override the direct MiniMax OpenAI-compatible API base URL. |
| `ATAGIA_OPENROUTER_BASE_URL` | _(unset)_ | Optional | Override OpenRouter API base URL. |
| `ATAGIA_OPENROUTER_SITE_URL` | `http://localhost` | Optional | `HTTP-Referer` header sent to OpenRouter for attribution. |
| `ATAGIA_OPENROUTER_APP_NAME` | `Atagia` | Optional | `X-Title` header sent to OpenRouter for attribution. |

### Inference access modes

Atagia defaults to `unrestricted`. Two opt-in modes constrain every
Atagia-owned completion, streaming, and embedding route before its transport
is built or called:

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_INFERENCE_ACCESS_MODE` | `unrestricted` | Optional | `unrestricted`, `local_only`, or `zero_cost`. |
| `ATAGIA_LOCAL_LLM_ENDPOINTS_FILE` | _(unset)_ | Required for `local_only`; optional for `zero_cost` | Absolute path to an immutable version-one JSON catalog of local OpenAI-compatible endpoints and their served models. |
| `ATAGIA_ZERO_COST_OPENROUTER_PROFILE` | _(unset)_ | Conditional | Must be `dedicated_free_tier_no_byok` before `zero_cost` can admit an external OpenRouter-free route. It is unnecessary for an all-local `zero_cost` configuration. |

Local model specs have the form
`local/<endpoint_id>/<served_model_id>[,thinking_level]`. The endpoint ID keeps
two machines serving the same model unambiguous, while the served model ID may
itself contain `/`. Credentials are named with `api_key_env` and read from that
environment variable; never put a secret in the catalog.

A single local endpoint catalog can be as small as:

```json
{
  "version": 1,
  "endpoints": [
    {
      "id": "desktop",
      "adapter": "openai_compatible",
      "base_url": "http://127.0.0.1:11434/v1",
      "chat_models": ["qwen3:8b", "qwen3-coder:30b"],
      "embedding_models": ["qwen3-embedding:4b"]
    }
  ]
}
```

Several LAN machines use the same schema. These addresses are examples; local
catalog hosts must be literal loopback, RFC1918 IPv4, or IPv6 ULA addresses and
must include a port:

```json
{
  "version": 1,
  "endpoints": [
    {
      "id": "generation_gpu",
      "adapter": "openai_compatible",
      "base_url": "http://10.0.0.20:8000/v1",
      "api_key_env": "GENERATION_GPU_API_KEY",
      "chat_models": ["acme/assistant-32b", "acme/assistant-70b"],
      "embedding_models": []
    },
    {
      "id": "embedding_node",
      "adapter": "openai_compatible",
      "base_url": "http://10.0.0.21:8080/v1",
      "chat_models": [],
      "embedding_models": ["acme/embed-4b"]
    }
  ]
}
```

For an entirely local runtime, route every enabled surface to catalog-backed
specs. Embeddings need their own local spec only when enabled:

```env
ATAGIA_INFERENCE_ACCESS_MODE=local_only
ATAGIA_LOCAL_LLM_ENDPOINTS_FILE=/srv/atagia/local_llm_endpoints.json
ATAGIA_LLM_FORCED_GLOBAL_MODEL=local/generation_gpu/acme/assistant-32b
ATAGIA_EMBEDDING_BACKEND=sqlite_vec
ATAGIA_EMBEDDING_MODEL=local/embedding_node/acme/embed-4b
ATAGIA_EMBEDDING_DIMENSION=1536
```

`zero_cost` always admits the same catalog-backed local routes. With no
OpenRouter profile it remains all-local. To add an external completion route,
use only an exact OpenRouter `:free` variant or `openrouter/free`, an OpenRouter
free-tier key, the official canonical origin, and the dedicated profile:

```env
ATAGIA_INFERENCE_ACCESS_MODE=zero_cost
ATAGIA_LOCAL_LLM_ENDPOINTS_FILE=/srv/atagia/local_llm_endpoints.json
ATAGIA_LLM_INGEST_MODEL=local/generation_gpu/acme/assistant-32b
ATAGIA_LLM_RETRIEVAL_MODEL=local/generation_gpu/acme/assistant-32b
ATAGIA_LLM_CHAT_MODEL=openrouter/openrouter/free
ATAGIA_OPENROUTER_API_KEY=replace-with-a-dedicated-free-tier-key
ATAGIA_ZERO_COST_OPENROUTER_PROFILE=dedicated_free_tier_no_byok
```

Startup checks OpenRouter's `GET /api/v1/key` result for
`data.is_free_tier=true`; custom OpenRouter base URLs fail closed. Admitted
requests receive policy-owned zero maximum prices and disabled provider
fallback. The named profile is also an operator declaration that the dedicated
OpenRouter account or workspace has no BYOK providers: Atagia cannot inspect
that account invariant atomically. External embeddings and all other external
providers remain denied.

When OPF privacy filtering is enabled in either restricted mode, its primary
and fallback URLs must both be literal local URLs. Version one has no external
zero-cost OPF route.

Atagia-bench and LoCoMo accept the same
`--inference-access-mode`, `--local-llm-endpoints-file`, and
`--zero-cost-openrouter-profile` options. Their default judge is not an
admissible restricted route, so select an admitted judge explicitly, for
example `--judge-model local/generation_gpu/acme/assistant-32b`. Judge and
engine routes are audited together before provider construction.

These modes govern provider HTTP performed by the Atagia runtime and its
associated provider charges. They do not govern a host application's own
answer-model call, an HTTP client's remote Atagia sidecar, local electricity or
hardware costs, or competitor systems called outside Atagia. Use network
firewall or process-level egress controls when the whole machine must be unable
to reach external services.

## 4. LLM model selection

Model specs are provider-qualified: `provider/model[,thinking_level]`, e.g.
`anthropic/claude-sonnet-4-6`, `kimi/kimi-k2.7-code`,
`minimax/MiniMax-M3`, or `openrouter/minimax/minimax-m3`.
Resolution order per component: forced-global -> component override -> enabled
finite-decision route (supported components only) -> inherited component ->
category override -> built-in default.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_LLM_FORCED_GLOBAL_MODEL` | _(unset)_ | Optional | Force a single model for every completion component (highest priority). |
| `ATAGIA_LLM_INGEST_MODEL` | _(unset)_ | Optional | Category override for all ingest-side components. |
| `ATAGIA_LLM_RETRIEVAL_MODEL` | _(unset)_ | Optional | Category override for all retrieval-side components. |
| `ATAGIA_LLM_CHAT_MODEL` | _(unset)_ | Optional | Category override for the chat component. |
| `ATAGIA_LLM_FINITE_DECISIONS_ENABLED` | `false` | Optional | Explicitly opt supported finite-choice components into the common decision route. When enabled without an ordinary decision model, the route is `typesafe/jev-latest`. |
| `ATAGIA_LLM_FINITE_DECISION_MODEL` | _(unset)_ | Optional | Ordinary provider-qualified LLM used by all supported finite-choice components when the opt-in switch is enabled. TypeSafe specs are rejected here; leave it unset for Jev. |
| `ATAGIA_LLM_MODEL__<COMPONENT_ID>` | _(unset)_ | Optional | Per-component override. `<COMPONENT_ID>` is one of the IDs listed below, uppercased. |

### Component IDs (for `ATAGIA_LLM_MODEL__<COMPONENT_ID>`)

Ingest: `EXTRACTOR`, `DATE_RESOLUTION`, `TEXT_CHUNKER`, `COMPACTOR`, `SUMMARY_PRIVACY_JUDGE`,
`SUMMARY_PRIVACY_REFINER`, `BELIEF_REVISER`, `CONTRACT_PROJECTION`,
`GRAPH_PROJECTION`, `CONSEQUENCE_BUILDER`, `CONSEQUENCE_DETECTOR`,
`CONSEQUENCE_GATE`, `CONSEQUENCE_SENTIMENT`, `CONSEQUENCE_LINK`,
`TOPIC_WORKING_SET`, `CONSENT_CONFIRMATION`, `INTENT_CLASSIFIER`,
`EXTRACTION_WATCHDOG`, `INITIAL_CONTEXT_PACKAGE_CURATION`, `EXPORT_ANONYMIZER`.

Retrieval: `NEED_DETECTOR_NEEDS`, `NEED_DETECTOR_LANGUAGE`,
`NEED_DETECTOR_MEMORY`, `NEED_DETECTOR_EXACT`, `NEED_DETECTOR_SHAPE`,
`NEED_DETECTOR_FACETS`, `NEED_DETECTOR_CALLBACK`,
`NEED_DETECTOR_SEARCH_WORDS`, `NEED_DETECTOR_SEARCH_WORDS_OTHER_LANGUAGE`,
`COVERAGE_EXPANDER`, `APPLICABILITY_SCORER`, `APPLICABILITY_RELEVANCE`, `CONTEXT_STALENESS`,
`METRICS_COMPUTER`.

Chat: `ANSWER_POSTCONDITION`, `CHAT`.

### Calendar date resolution

`DATE_RESOLUTION` defaults to `openrouter/openai/gpt-6-luna,low`. Override it with
`ATAGIA_LLM_MODEL__DATE_RESOLUTION`; the usual forced-global, component, category,
intimacy, and inference-access rules still apply. This setting does not change
reasoning for other extraction or retrieval calls. OpenRouter date requests
explicitly disable provider fallback. The ordinary completion-token floor stays
in force: the card asks for 1024 tokens and the client currently sends 8192.

The date card extracts one operation from text and the source message's calendar
date. Python applies calendar months and years, clamps the end of the month,
estimates a remaining fractional month at 30 days, and rounds fractional days
to the nearest day with ties away from zero. `ref` always refers to the source
message date. Temporal classification remains a separate finite decision.

The memory payload stores a versioned `date_resolution`, the source reference,
and a text hash. Completed `exact`, `uncertain`, and `unknown` results are reused
on retrieval. Uncertainty describes the wording, not a numeric confidence or a
prediction that an event will occur. `analyze` is `pending_analysis`; missing
annotations are unprocessed and mismatched text/anchors are stale. Retrieval
never turns those states into completed unknowns or makes another date call.

Intervals retain separate boundary, clock, UTC-offset, and calendar-period
interpretation. Each normalized endpoint is attributed through an exact
`source_quote` from an original message; timing wording may inherit explicitly
written shared calendar fields. Rewritten candidates and prior-chunk summaries
are not literal source evidence. Quotes may match the original text or its exact
HTML-escaped prompt representation; a unique match is stored as the original
substring. Literal entity text is preserved, and decoding is never repeated.
Ambiguous source attribution remains pending, including when raw and escaped
representations identify different sources or original substrings.
For an explicitly written whole month or year, typed `calendar_period` fields
let Python construct the first and last instants without inventing a point date.
Relative timing still uses the independent date resolver and source message date.

Python builds timestamps only when a supported bound and offset are available.
Approximate representative dates do not become exact expiration timestamps.
Context evidence prefers the actual validity interval, retaining open bounds,
offsets and uncertainty. A range's last day is not the date of the whole state;
the separate point annotation remains available for audit. Stale or inconsistent
interval metadata does not silently fall back to that point.

There is no automatic real-data backfill. Existing memories, stale annotations,
or pending analysis can be handled with the existing admin conversation rebuild
(`POST /v1/admin/rebuild/conversation/{conversation_id}`) under normal admin
and user partition controls. It clears managed derived state and replays stored
messages through ingestion, so first back up and select a bounded conversation.
Re-inserting a duplicate message is not a date backfill. If the source remains
incomplete, a rebuild may leave analysis pending again; it does not invent a
missing anchor or retry indefinitely.

### TypeSafe finite-choice decisions (opt-in)

```dotenv
ATAGIA_TYPESAFE_API_KEY=your-typesafe-key
ATAGIA_LLM_FINITE_DECISIONS_ENABLED=true
```

To use the same finite-decision components without TypeSafe access, select any
ordinary provider-qualified model instead:

```dotenv
ATAGIA_LLM_FINITE_DECISIONS_ENABLED=true
ATAGIA_LLM_FINITE_DECISION_MODEL=local/desktop/decision-model
```

The switch is authoritative. When it is false, a TypeSafe per-component
override fails configuration validation instead of remaining silently active.
Setting `ATAGIA_LLM_FINITE_DECISION_MODEL` while the switch is false also fails.
The shared decision-model setting accepts ordinary LLMs only; leave it unset to
select Jev. Forced-global and explicit component overrides retain their higher
precedence. The finite-decision route outranks category models, while unrelated
components continue to use their category or built-in models. A supported
component can therefore opt out of the shared route with its own ordinary model
override, or explicitly select Jev while the switch is enabled.

The narrow extraction decisions have independent component overrides.
`EXTRACTION_TEMPORAL_TYPE` uses the common finite-decision route when enabled;
the other extraction subcards below inherit `EXTRACTOR` unless overridden.
Candidate generation, belief keys and values, member names, and interval-shape
extraction use the extractor model, whose built-in default is GPT-6-Luna.
Calendar date operations use the separate `DATE_RESOLUTION` component. Claim-key
equivalence retains the existing `INTENT_CLASSIFIER` finite-decision route.

| Component suffix | Decision | Remaining generative work |
|---|---|---|
| `EXTRACTION_KIND` | One memory kind | Candidate text |
| `EXTRACTION_SCOPE` | One allowed scope; a single allowed scope needs no call | None |
| `EXTRACTION_CONFIDENCE` | Source-support score in [0,1] | None |
| `EXTRACTION_EVIDENCE_SUPPORT` | Direct, contextual, inferred, weak, or absent support | None |
| `EXTRACTION_PRESERVE_VERBATIM` | Whether exact wording must be retained | Candidate language list |
| `EXTRACTION_TEMPORAL_TYPE` | Permanent, bounded, event-triggered, ephemeral, unknown, or absent | Intervals when required |
| `EXTRACTION_MEMBER_IDENTITY` | One canonical identity per member; native choices use the extracted-name catalog | Member list; LLM identity generation; native identities absent from the catalog |

`EXTRACTION_CONFIDENCE` asks an ordinary LLM for one continuous source-support
number using the existing simple extraction prompt. Jev receives a five-anchor
rubric and returns a weighted Score on [0,4], divided by four. TypeSafe's
certainty about its answer is separate from this memory confidence. Existing
memory activation thresholds do not change.

`EXTRACTION_MEMBER_IDENTITY` keeps one simple canonical-name generation request
per member on LLM routes. Native TypeSafe choices compare each member against
the names already extracted for that candidate. A valid `not_listed` choice
requests a canonical name from the extractor model; it does not substitute a
model after an error. The full source context remains available to both routes.

`EXTRACTION_EVIDENCE` selects original source references and defaults to
`openrouter/openai/gpt-6-luna`, independently of an `EXTRACTOR` override.
Forced-global, explicit component and ingest-category settings retain their
normal precedence. GPT-6-Luna defaults to reasoning effort `none`; an explicit
thinking suffix can select another supported effort. Support, literal
preservation and the candidate language list remain separate cards.

An explicit `typesafe/jev-latest` override with finite decisions enabled selects
source boundaries through native choices. The selector reuses the support
assessment and candidate text; it does not extract or classify them again. Both
routes receive the source and relevant conversation context. Code copies the
quotation from source coordinates after validating the catalog and source hash;
the model does not reproduce quote text or count characters. Long sources use
block and anchor stages. Native transport splits ready questions when necessary,
but does not truncate source context; an input beyond its safe context limit
fails before dispatch. All effective requests count toward cost and concurrency.
Invalid decisions fail explicitly without silent model fallback.

Query and answer language decisions can be overridden independently through
`NEED_DETECTOR_QUERY_LANGUAGE` and `NEED_DETECTOR_ANSWER_LANGUAGE`. They inherit
`NEED_DETECTOR_LANGUAGE` otherwise. Answer language still receives the query
language result and the existing profile/preference context.

`ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE=direct|selective` controls existing-topic
content updates and defaults to `selective`. Selective mode offers one independent
keep/clear/regenerate decision per field; a required title cannot be cleared.
Its component suffixes are `TOPIC_TITLE_DECISION`, `TOPIC_SUMMARY_DECISION`,
`TOPIC_GOAL_DECISION`, `TOPIC_QUESTIONS_DECISION` and `TOPIC_DECISIONS_DECISION`.
With finite decisions enabled they use the common decision route (Jev unless an
ordinary decision model is selected). With the switch off they inherit
`TOPIC_WORKING_SET`; explicit component overrides take precedence in both cases.
New topics bypass these filters. `TOPIC_WORKING_SET` defaults to
`openrouter/openai/gpt-6-luna`. A regenerated field uses that topic model with the
full context. Keeping or clearing avoids that generation, but filtering a field that
needs regeneration adds a decision. Choose this mode based on the complete
workflow's cost, latency and quality, rather than its filter alone. Setting the
update mode to `direct` explicitly skips these field decisions.

`CONTEXT_STALENESS` follows the same route: with finite decisions enabled and no
component override, it uses Jev's native `reuse`/`refresh` choice after cache
identity, policy, revision, age, and deterministic checks. An explicit
`ATAGIA_LLM_MODEL__CONTEXT_STALENESS=openrouter/google/gemini-3.1-flash-lite`
keeps the conventional LLM signal detector for that component. The switch
remains off by default, so installations that have not enabled finite decisions
retain their existing context-cache model. Missing provider keys and invalid
native responses fail explicitly; there is no automatic fallback model.

`APPLICABILITY_RELEVANCE` inherits `APPLICABILITY_SCORER` when neither a component
override nor the enabled common decision route applies. The forced-global
setting still takes precedence, so leave it
unset to use different models for relevance and dates. Few-shot examples
continue to use the existing `APPLICABILITY_SCORER` examples setting.

TypeSafe uses the native [System One API](https://docs.typesafe.ai/api), not an
OpenAI-compatible chat endpoint. Each candidate receives one independent choice
among Atagia's existing relevance labels. Candidate retrieval, numeric label
weights, and ranking retain their existing roles. Retrieval consumes persisted
date annotations without dispatching a date card; extraction and answer
generation keep their own models.

The seven supported need-detection cards use closed choices:

| Component suffix | Native decisions |
|---|---|
| `MEMORY`, `EXACT`, `SHAPE` | Memory dependence, exact recall, and answer shape, separately. |
| `LANGUAGE` | Two separate requests: query language and answer language. Each selects from the existing complete ISO 639-1 catalog plus `unknown`, mapped to absent guidance. Codes identify languages (`en`, `it`), not countries (`US`, `IT`). A current answer-language request outranks a stored preference. |
| `NEEDS` | Independent yes/no membership questions for the policy's enabled need types. Multiple matches and no matches are valid. An empty enabled set needs no provider call. |
| `FACETS` | Independent yes/no membership questions for the ten exact-detail categories. Several categories may apply; this is not a single-label choice. |
| `CALLBACK` | Whether the user refers to an earlier assistant answer or recommendation. |

`INTENT_CLASSIFIER` also supports separate yes/no decisions for explicit durable
user statements and claim-key equivalence. These decisions do not generate
reasoning text. Identical claim keys still return true mechanically without I/O.

The common finite route covers applicability relevance; the seven closed-choice
need cards; the two intent-classifier decisions; consequence gate, sentiment,
and link; and the separate context-reuse decision. No unrelated semantic tasks
share a multi-task inference. Independent
membership questions within one card family may use one API batch. Consequence
action, outcome, and language stay on `CONSEQUENCE_DETECTOR`. Search words and
cross-language aliases, free-text extraction, language-profile synthesis, date
resolution, and answer generation remain on generative models.

In a frozen synthetic comparison of 12 semantic context-reuse cases repeated ten
times, `jev-1.13.0` matched the expected decision 110/120 times versus 90/120
for the conventional Luna route. All ten Jev errors were one ambiguous-reference
case that asked for "the other one" after two possible referents. That weakness
concerns ambiguity, not necessarily an old cache entry. This small result does
not establish general production accuracy. In that case Jev reused context
when the expected action was refresh. The per-component override above remains
available while this limitation is tracked.

Enabling a route sends that card's required message/context/profile excerpts
to TypeSafe. Use synthetic or explicitly authorized data during evaluation;
enabling local configuration is not a deployment or a memory-quality guarantee.

TypeSafe is not accepted as a global, category-wide, chat, JSON-rescue, or
intimacy-fallback model. Typed requests have no text parsing, JSON repair, or
automatic chat-model fallback. Authentication and malformed responses fail
immediately; transient transport errors use the existing shared retry policy.
The common decision override and TypeSafe are off by default; the same decision
tasks still run through their ordinary component models. Store credentials in
the environment or your ignored local `.env`, never in tracked files.
TypeSafe is an external, metered provider: both `local_only` and `zero_cost`
reject its routes at startup and before calls. Free output tokens do not make
the input-token charges eligible for `zero_cost`.

### Card prompt examples (few-shot demonstrations)

Card prompts (need detection, memory extraction, applicability scoring, topic
working set, language profile, consequence detection) include a few-shot
demonstration block by default. Concrete examples reliably help small/local
models follow the output format and decision, but can hurt larger or reasoning
models, so the block is toggleable per component without maintaining a second
prompt set. The instruction and output-format spec are always sent; only the
demonstration block is gated.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_CARD_EXAMPLES_ENABLED` | `true` | Optional | Global default for including the examples block in card prompts. |
| `ATAGIA_LLM_EXAMPLES__<COMPONENT_ID>` | _(unset)_ | Optional | Per-component override (`true`/`false`). Wins over the global default. Same component IDs as `ATAGIA_LLM_MODEL__<COMPONENT_ID>`; the need-detection cards use the per-card component IDs `NEED_DETECTOR_NEEDS`, `NEED_DETECTOR_LANGUAGE`, `NEED_DETECTOR_MEMORY`, `NEED_DETECTOR_EXACT`, `NEED_DETECTOR_SHAPE`, `NEED_DETECTOR_FACETS`, `NEED_DETECTOR_CALLBACK`, `NEED_DETECTOR_SEARCH_WORDS`, and `NEED_DETECTOR_SEARCH_WORDS_OTHER_LANGUAGE`. |

### Structured-output repair and rescue

All structured LLM calls first pass through the shared JSON cleanroom and
schema validation. If validation still fails, Atagia can make a bounded
corrective retry with the same model. If that also fails, an optional rescue
model can be enabled for structured-output tasks only. This is intended for
fast local/dev routing: use cheap/fast models first, then escalate only when a
JSON contract is actually stuck.

The rescue path is off by default. Its configured default model is direct
Anthropic Opus 4.7 because local May 2026 benchmark runs showed stable
structured verdicts there, while the OpenRouter judge route produced schema/JSON
technical failures. When enabled, the rescue model is provider-qualified and
requires the corresponding provider API key at startup. The failing output
excerpt and validation details are sent to the retry/rescue call, so keep this
disabled unless the configured model/provider is acceptable for the component's
data.

Repair observability is explicit: rescue escalation emits a warning log, retry
and rescue requests carry `atagia_structured_output_*` metadata, and benchmark
reports aggregate calls under
`config.llm_call_summary.structured_output_repair`.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_LLM_STRUCTURED_OUTPUT_RETRY_ATTEMPTS` | `1` | Optional | Same-model corrective retries after cleanroom/schema validation fails. |
| `ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_ENABLED` | `false` | Optional | Enable final escalation to the configured rescue model after same-model retries fail. |
| `ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL` | `anthropic/claude-opus-4-7` | Optional | Provider-qualified rescue model, for example `anthropic/claude-opus-4-7` or `openai/gpt-5.5`. Required provider key only matters when rescue is enabled. |

### Provider dispatch

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_LLM_MAX_CONCURRENT_REQUESTS_PER_PROVIDER` | `4` | Optional | Maximum in-flight requests to each provider across tasks using one engine's LLM client. Completions, embeddings, and streams share this limit; separate processes or engine clients have independent limits. |

### Answer context envelope

Answer-time prompts use one structural input envelope by default. The envelope
allocates the full global budget across instructions, the current turn,
retrieved context, and recent transcript; it does not force empty filler when a
section has less useful material than its allocation. The current default is
the retained-replay calibrated 8k budget.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_CONTEXT_ENVELOPE_BUDGET_TOKENS` | `8192` | Optional | Global answer-input envelope budget used to derive section budgets for retrieved context and recent transcript. |
| `ATAGIA_CONTEXT_ENVELOPE_RATIOS` | `instructions=0.10,current_turn=0.03,retrieved_context=0.67,recent_transcript=0.20` | Optional | Section allocation ratios. Accepts either a JSON object or comma-separated `key=value` pairs; values are normalized before allocation. |

---

## 5. Response modes and adaptive retrieval

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_RESPONSE_MODE` | `normal` | Optional | Default per-turn retrieval scheduling. `normal` runs the full retrieval pipeline synchronously. `fast` skips retrieval entirely and answers from the prepared context (interaction contract, recent transcript, topic working set, initial context package). `smart_fast` answers like `fast` on the current turn and runs a full retrieval in the background to warm the context for the next turn. |
| `ATAGIA_ADAPTIVE_RETRIEVAL` | `true` | Optional | Enables the adaptive retrieval gate. When on, the need detector classifies each turn's memory dependence (`personal`, `conversation`, `world`, `mixed`); turns classified `world` or `conversation` skip the expensive retrieval stages and answer from the prepared context, while `personal`/`mixed` turns run full retrieval. Adds no extra LLM calls: the classification is part of the existing need-detection call. Set to `false` to force full retrieval on every turn (the classification is still computed and recorded in diagnostics). |

How the two settings combine:

- `normal` + adaptive on: the gate decides per turn whether to retrieve.
  Uncertain or degraded turns always retrieve.
- `normal` + adaptive off: full retrieval every turn. The classification is
  still computed and recorded in diagnostics (shadow mode), with no behavior
  change.
- `fast`: never retrieves; the adaptive flag is a documented no-op
  (diagnostics report `not_applicable`).
- `smart_fast` + adaptive on: the current turn is unchanged; the background
  warm is skipped when the gate classifies the turn as not memory-dependent.

Both settings accept per-request overrides: the `response_mode` and
`adaptive_retrieval` fields on chat/context API requests and the library-mode
`chat(...)`/`get_context(...)` calls, and the `X-Atagia-Response-Mode` and
`X-Atagia-Adaptive-Retrieval` headers on the OpenAI-compatible proxy. A
gate-skipped turn never writes the context cache, and memory extraction still
runs for every turn regardless of the gate decision.

---

## 6. Embeddings

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_EMBEDDING_BACKEND` | `none` | Optional | Embedding backend. `none` runs FTS-only retrieval; `sqlite_vec` enables hybrid vector search. |
| `ATAGIA_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Optional | Provider-qualified embedding model spec. |
| `ATAGIA_EMBEDDING_DIMENSION` | `1536` | Optional | Embedding vector dimension. Must match the configured embedding model's output dimension. |
| `ATAGIA_EMBEDDING_VECTOR_LIMIT_CAP` | `50` | Optional | Maximum vector candidates returned by sqlite-vec per query before applicability scoring. |
| `ATAGIA_EMBEDDING_SEARCH_OVERFETCH_MULTIPLIER` | `4` | Optional | Multiplier applied to the requested limit when over-fetching vector candidates before fusion. |

---

## 7. Intimacy fallback policy

Atagia supports a category-level and component-level fallback model that
takes over when the primary model returns a provider policy block or refusal
on intimate content. The fallback is invoked only after the primary attempt
fails with a policy-block or refusal signal (HTTP refusal codes, finish
reasons containing `refusal`, or the OpenAI `response:refusal` marker). When
the fallback is used, the request metadata records:

- `atagia_intimacy_fallback_used = True`
- `atagia_intimacy_primary_model = <primary model spec>`
- `atagia_intimacy_primary_error_class = <exception class>`
- `atagia_intimacy_primary_error_reason = <safe error label>`

With `ATAGIA_LLM_INTIMACY_PROACTIVE_ROUTING_ENABLED=true`, requests carrying
known-intimate metadata (non-ordinary `intimacy_boundary` on the topic
working set or in the resolved policy) are routed directly to the configured
intimacy model without first probing the primary model. In that case the
request metadata records `atagia_intimacy_proactive_route = True`.

Resolution order for the intimacy fallback per component: explicit
`atagia_intimacy_fallback_model` request override -> component intimacy
override -> category intimacy override.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_LLM_INTIMACY_INGEST_MODEL` | _(unset)_ | Optional | Category-level intimacy fallback model for ingest-side components. |
| `ATAGIA_LLM_INTIMACY_RETRIEVAL_MODEL` | _(unset)_ | Optional | Category-level intimacy fallback model for retrieval-side components. |
| `ATAGIA_LLM_INTIMACY_MODEL__<COMPONENT_ID>` | _(unset)_ | Optional | Per-component intimacy fallback override. Same component IDs as in section 4. |
| `ATAGIA_LLM_INTIMACY_PROACTIVE_ROUTING_ENABLED` | `false` | Optional | Skip the primary model and route known-intimate requests directly to the intimacy fallback model. |

---

## 8. Chunking

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_DISABLE_CHUNKING_EXTRACTION` | `false` | Optional | When true, bypass two-level chunking during extraction and pass full text directly to the extractor. |
| `ATAGIA_CHUNKING_EXTRACTION_THRESHOLD_TOKENS` | `2048` | Optional | Token threshold above which the extractor activates chunked extraction. Must be positive. |

---

## 9. Initial context package rollout

The initial context package is a prepared, query-independent context layer.
SQLite remains canonical truth, and normal turns still run query-specific
retrieval. These controls are rollout and validation switches; they are not
fast-mode answer controls.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_INITIAL_CONTEXT_PACKAGE_READ_ENABLED` | `true` | Optional | Enable prompt-time reads of fresh prepared baseline/conversation packages. A disabled read falls open to the live context path. |
| `ATAGIA_INITIAL_CONTEXT_PACKAGE_REFRESH_ENABLED` | `true` | Optional | Enable background materialization of package refresh jobs. When false, existing affected packages are still marked stale, but queued or future refresh jobs do not build new active packages. |
| `ATAGIA_INITIAL_CONTEXT_PACKAGE_PROMPT_MAX_TOKENS` | `900` | Optional | Maximum prompt budget considered for the rendered prepared package block at turn time. Must be positive. |
| `ATAGIA_INITIAL_CONTEXT_PACKAGE_PROFILE_MAX_TOKENS` | `700` | Optional | Maximum token budget for the prepared memory profile block during package materialization. Must be positive. |
| `ATAGIA_INITIAL_CONTEXT_PACKAGE_TOTAL_MAX_TOKENS` | `2200` | Optional | Maximum token budget for the full materialized package body. Must be positive. |

Text-free rollout prompt-diff artifacts store prompt hashes, token estimates,
package status, and request-path counters; they do not store raw prompt text.

---

## 10. Worker circuit breaker

Brief reference; see
[`docs/HOST_SIDECAR_INTEGRATION.md`](HOST_SIDECAR_INTEGRATION.md) for the
operational behavior and recovery flow.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_WORKER_CIRCUIT_BREAKER_ENABLED` | `true` | Optional | Master toggle for the worker-level circuit breaker. |
| `ATAGIA_WORKER_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `20` | Optional | Failure count within the window that trips the breaker. Must be positive. |
| `ATAGIA_WORKER_CIRCUIT_BREAKER_WINDOW_SECONDS` | `180` | Optional | Sliding window length (seconds) used to evaluate the failure threshold. |
| `ATAGIA_WORKER_CIRCUIT_BREAKER_MIN_FAILURE_RATIO` | `0.8` | Optional | Minimum failure-to-attempt ratio within the window required to trip. Range `[0.0, 1.0]`. |

---

## 11. Debug and observability

These are intended for local development and benchmark diagnostics. Keep
disabled in production. When enabled, raw prompt/response artifacts are
written to disk under `ATAGIA_DEBUG_LLM_IO_DIR`, one JSON file per LLM call
with optional raw request/response bodies.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_DEBUG` | `false` | Optional | Generic debug toggle for verbose logging. |
| `ATAGIA_DEBUG_LLM_IO` | `false` | Optional | Persist every LLM call to disk for inspection. |
| `ATAGIA_DEBUG_LLM_IO_DIR` | `./data/llm_debug` | Optional | Directory where LLM IO artifacts are written. |
| `ATAGIA_DEBUG_LLM_IO_PURPOSES` | _(empty)_ | Optional | Comma-separated allowlist of purposes to record. Empty means record all. |
| `ATAGIA_DEBUG_LLM_IO_RAW` | `false` | Optional | Also persist raw provider request/response payloads alongside the structured artifact. |
| `ATAGIA_DEBUG_LLM_IO_MAX_CHARS` | `50000` | Optional | Maximum characters per recorded field before truncation. |

Diagnostic capture is a separate, opt-in local recorder. It stores rendered
model inputs, typed options and schemas, resolved provider attempts, responses,
stream fragments, source snapshots, and selected operation effects. This can
include conversation content. It does not upload captures or make extra model
calls. Keep the directory private and remove capture directories manually when
they are no longer needed. The directory must already exist and be writable
when capture is enabled; startup fails if it is unavailable. Each session has
`manifest.json`, `events.jsonl`, and SHA-256 addressed `blobs/`. A complete
manifest is required for replay. If a size or write limit is hit, the manifest
is marked failed and the provider result is still returned normally.

| Variable | Default | Status | Purpose |
|---|---|---|---|
| `ATAGIA_DIAGNOSTIC_CAPTURE_ENABLED` | `false` | Optional | Enable local diagnostic capture for this process. |
| `ATAGIA_DIAGNOSTIC_CAPTURE_DIR` | `./data/diagnostic_captures` | Optional | Existing private parent directory for per-session captures. |
| `ATAGIA_DIAGNOSTIC_CAPTURE_MAX_BLOB_BYTES` | `1048576` | Optional | Maximum bytes for one exact content block; excess marks capture failed. |
| `ATAGIA_DIAGNOSTIC_CAPTURE_MAX_SESSION_BYTES` | `104857600` | Optional | Maximum bytes for one session; excess marks capture failed. |

---

## 12. Service mode

Service mode runs Atagia as a FastAPI HTTP service. The library mode trusts
the caller's `user_id`; the service mode requires an API key.

`ATAGIA_SERVICE_MODE`, `ATAGIA_SERVICE_API_KEY`, `ATAGIA_ADMIN_API_KEY`,
`ATAGIA_ALLOW_INSECURE_HTTP` and `ATAGIA_WORKERS_ENABLED` configure the HTTP
service only. The library-mode engine (`Atagia(...)`) pins all five on its own
authority and ignores them: it is not a service, it runs its own workers, and it
talks over loopback. Every other variable in this reference reaches the runtime
in both modes. The effective-settings report tags these five `engine_override`
in library mode so a run never claims to have honored them.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_SERVICE_MODE` | `false` | Optional | Enable HTTP service mode. The shipped `.env.example` sets this to `true`. |
| `ATAGIA_SERVICE_API_KEY` | _(unset)_ | Required for service mode | Shared server-side credential required by every non-admin HTTP endpoint; never expose it to browser code or another untrusted client. |
| `ATAGIA_ADMIN_API_KEY` | _(unset)_ | Required for service mode | Distinct API key required at service startup and by admin endpoints. It must not equal `ATAGIA_SERVICE_API_KEY`; the client SDK uses it for admin operations. |
| `ATAGIA_BASE_URL` | _(unset)_ | Optional | Base URL used by the client SDK to reach the service. |
| `ATAGIA_ALLOW_INSECURE_HTTP` | `false` | Optional | Local-only escape hatch that allows non-TLS HTTP outside loopback. Keep `false` in production. |
| `ATAGIA_CORS_ALLOWED_ORIGINS` | _(empty)_ | Optional | Comma-separated origin allowlist for browser-based clients (e.g. `http://127.0.0.1:8000`). |
| `ATAGIA_WORKERS_ENABLED` | `false` | Optional | Start background workers alongside the HTTP service. The shipped `.env.example` sets this to `true`. |
| `ATAGIA_PROXY_MODEL_ID` | `atagia-memory-proxy` | Optional | Visible model id surfaced to OpenAI-compatible proxy clients. |
| `ATAGIA_PROXY_UPSTREAM_MODEL` | _(unset)_ | Optional | Provider-qualified upstream model used by the proxy. Falls back to the configured chat model when unset. |
| `ATAGIA_PROXY_DEFAULT_MODE` | _(unset)_ | Optional | Default assistant mode applied by the proxy when the client does not send one. |
| `ATAGIA_PROXY_MAX_OUTPUT_TOKENS` | `8192` | Optional | Server ceiling for externally visible proxy answers. If the client supplies `max_tokens`, `max_completion_tokens`, or both, the effective provider limit is the smallest positive client value and this server ceiling. Internal extraction/planning calls retain their separate output headroom. |
| `ATAGIA_REQUEST_MAX_BODY_BYTES` | `33554432` (32 MiB) | Optional | Maximum encoded HTTP request body. Enforced for both `Content-Length` and chunked bodies. |
| `ATAGIA_REQUEST_MAX_MESSAGE_TEXT_BYTES` | `262144` (256 KiB) | Optional | Maximum UTF-8 byte length of one external message text or typed text block. |
| `ATAGIA_REQUEST_MAX_ATTACHMENTS` | `16` | Optional | Maximum attachment or typed multimodal attachment-block count per request. |
| `ATAGIA_REQUEST_MAX_ATTACHMENT_DECODED_BYTES` | `10485760` (10 MiB) | Optional | Maximum decoded bytes for one attachment. Base64 length is validated before decoding. |
| `ATAGIA_REQUEST_MAX_ATTACHMENTS_DECODED_BYTES` | `20971520` (20 MiB) | Optional | Maximum decoded bytes across all attachments in one request. Must be at least the per-attachment limit. |
| `ATAGIA_REQUEST_MAX_METADATA_BYTES` | `65536` (64 KiB) | Optional | Maximum UTF-8 byte length of compact serialized request or attachment metadata. |

Body and decoded-size overruns return `413`. Invalid base64 or other bounded
payload-structure failures return `422`. These limits apply to both the direct
chat/sidecar routes and the OpenAI-compatible proxy before provider execution
or artifact persistence.

---

## 13. MCP server

Variables read only by the MCP server entry point (`atagia-mcp`). These are
typically set in the host's MCP server config (e.g. Claude Desktop's
`mcp.json`).

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_USER_ID` | _(unset)_ | Required | Stable user identifier for the MCP session. The server refuses to start without it. |
| `ATAGIA_PLATFORM_ID` | _(unset)_ | Required | Stable platform/app identifier for the MCP session. The server refuses to start without it. |
| `ATAGIA_USER_PERSONA_ID` | _(unset)_ | Optional | User persona coordinate for the MCP session. |
| `ATAGIA_CHARACTER_ID` | _(unset)_ | Optional | Character/presence coordinate for the MCP session. |
| `ATAGIA_EMBODIMENT_ID` | _(unset)_ | Optional | Embodiment coordinate for the MCP session. |
| `ATAGIA_REALM_ID` | _(unset)_ | Optional | Realm coordinate for the MCP session. |
| `ATAGIA_CONVERSATION_ID` | _(unset)_ | Optional | Default conversation id for the MCP session. |
| `ATAGIA_INCOGNITO` | `false` | Optional | Run the MCP session in incognito mode (no persistence). |
| `ATAGIA_MCP_TRANSPORT` | `stdio` | Optional | MCP transport selector (`stdio`, `sse`, etc.). |

---

## 14. Sidecar bridge and client facade

Variables read by `SidecarBridgeConfig.from_env()` and the client facade. They
let host applications configure the transport and default memory coordinates
without hardcoding them.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_ENABLED` | `false` | Optional | Enable the fail-open sidecar bridge. |
| `ATAGIA_TRANSPORT` | `auto` | Optional | `auto`, `local`, or `http` for the sidecar/client transport. |
| `ATAGIA_BASE_URL` | _(unset)_ | Optional | HTTP service base URL. If set in auto mode, the client uses HTTP. |
| `ATAGIA_TIMEOUT_SECONDS` | `30` | Optional | Sidecar/client operation timeout. |
| `ATAGIA_MODE` | `personal_assistant` | Optional | Default retrieval profile for sidecar calls. |
| `ATAGIA_USER_PERSONA_ID` | _(unset)_ | Optional | Default persona coordinate for sidecar calls. |
| `ATAGIA_PLATFORM_ID` | _(unset)_ | Optional | Default platform coordinate for sidecar calls. |
| `ATAGIA_CHARACTER_ID` | _(unset)_ | Optional | Default character/project/prompt coordinate for sidecar calls. |
| `ATAGIA_ACTIVE_PRESENCE_ID` | _(unset)_ | Optional | Default active Presence coordinate when the host has one. |
| `ATAGIA_SPACE_ID` | _(unset)_ | Optional | Default Space coordinate for project/folder/capsule memory. |
| `ATAGIA_EMBODIMENT_ID` | _(unset)_ | Optional | Default Embodiment coordinate for body/device memory. |
| `ATAGIA_REALM_ID` | _(unset)_ | Optional | Default Realm coordinate for world/domain memory. |
| `ATAGIA_OPERATIONAL_PROFILE` | _(unset)_ | Optional | Default per-request runtime profile. |
| `ATAGIA_INCOGNITO` | `false` | Optional | Default incognito behavior for sidecar calls. |
| `ATAGIA_MEMORY_PRIVACY_MODE` | _(unset)_ | Optional | Default memory storage trust mode (`balanced`, `trusted_private`). |

---

## 15. Example environments

### 15.1. Minimal single-provider smoke env

All completion components routed to one provider via the forced-global
override. This is useful for smoke testing credentials and plumbing, but it
should not be used to judge retrieval quality because it replaces the
role-specific retrieval model.

```env
ATAGIA_OPENROUTER_API_KEY=your-openrouter-key
ATAGIA_LLM_FORCED_GLOBAL_MODEL=openrouter/deepseek/deepseek-v4-flash
ATAGIA_EMBEDDING_BACKEND=none
ATAGIA_SQLITE_PATH=./data/atagia.db
ATAGIA_DEBUG=true
```

### 15.2. Mixed-provider production env with intimacy fallback

Default routing (direct MiniMax M3 for ingest/compaction, OpenRouter Gemini
Flash-Lite for retrieval, OpenRouter DeepSeek v4 Flash for ordinary chat,
Anthropic for privacy/consent/export components, and direct Kimi K2.7 Code
for benchmark judging) plus OpenAI embeddings, service mode behind an API key,
and a per-component intimacy fallback for the extractor and compactor.

```env
ATAGIA_ANTHROPIC_API_KEY=your-anthropic-key
ATAGIA_MINIMAX_API_KEY=your-minimax-key
ATAGIA_KIMI_API_KEY=your-kimi-key
ATAGIA_OPENAI_API_KEY=your-openai-key
ATAGIA_OPENROUTER_API_KEY=your-openrouter-key
ATAGIA_EMBEDDING_BACKEND=sqlite_vec
ATAGIA_EMBEDDING_MODEL=openai/text-embedding-3-small
ATAGIA_EMBEDDING_DIMENSION=1536
ATAGIA_SERVICE_MODE=true
ATAGIA_SERVICE_API_KEY=your-service-key
ATAGIA_ADMIN_API_KEY=your-admin-key
ATAGIA_WORKERS_ENABLED=true
ATAGIA_LLM_INTIMACY_MODEL__EXTRACTOR=openrouter/z-ai/glm-4.6
ATAGIA_LLM_INTIMACY_MODEL__COMPACTOR=openrouter/z-ai/glm-4.6
```

### 15.3. Local OpenAI-compatible runtime benchmark env

After declaring an Ollama or other OpenAI-compatible server in the local
endpoint catalog above, select it through its explicit endpoint identity. This
keeps the same configuration valid when another LAN machine is added later.

```env
ATAGIA_INFERENCE_ACCESS_MODE=local_only
ATAGIA_LOCAL_LLM_ENDPOINTS_FILE=/srv/atagia/local_llm_endpoints.json
ATAGIA_LLM_FORCED_GLOBAL_MODEL=local/desktop/qwen3-coder:30b
ATAGIA_EMBEDDING_BACKEND=sqlite_vec
ATAGIA_EMBEDDING_MODEL=local/desktop/qwen3-embedding:4b
ATAGIA_EMBEDDING_DIMENSION=1536
```

Local model speed and quality vary widely by hardware and quantization;
measure on your own setup before drawing conclusions.
