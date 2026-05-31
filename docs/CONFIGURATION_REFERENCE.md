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
# fill in at least one provider API key
```

---

## 1. Required for default runtime

The default model selection routes ingest/retrieval components to OpenRouter
(stable Gemini Flash-Lite), ordinary chat answers to OpenRouter (DeepSeek v4
Flash), and privacy/consent/export-sensitive components to Anthropic (Claude
Sonnet 4.6). Benchmark CLIs also default to direct Anthropic Opus 4.7 for
judging so evaluator schema/JSON instability does not hide product or retrieval
failures. The minimum viable configuration therefore needs:

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_ANTHROPIC_API_KEY` | _(unset)_ | Required for default privacy/consent/export components and benchmark judge | API key for Anthropic Claude (used by `summary_privacy_judge`, `summary_privacy_refiner`, `consent_confirmation`, `export_anonymizer`, and benchmark default direct Opus judging). |
| `ATAGIA_OPENROUTER_API_KEY` | _(unset)_ | Required for default ingest/retrieval/chat components | API key for OpenRouter (used by every default ingest/retrieval component and by the default `chat` component). |

To run all completion components on a single provider instead, set
`ATAGIA_LLM_FORCED_GLOBAL_MODEL` (see [LLM model selection](#4-llm-model-selection)) and
provide only that provider's API key.

For quality and cost balance, prefer role-specific routing over a single
forced-global model. The defaults route retrieval and ingest intelligence to
`openrouter/google/gemini-3.1-flash-lite`, cheap ordinary-answer generation to
`openrouter/deepseek/deepseek-v4-flash`, and privacy/consent/export-sensitive
components to `anthropic/claude-sonnet-4-6`. For local experiments, point an
OpenAI-compatible base URL at an Ollama route such as `openai/qwen3-coder:30b`.
Use the stable `openrouter/google/gemini-3.1-flash-lite` slug, not the
deprecated `-preview` endpoint.

---

## 2. Storage

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_SQLITE_PATH` | `./data/atagia.db` | Optional | Path to the SQLite database used as the single source of truth. |
| `ATAGIA_DB_PATH` | `atagia.db` (MCP) | Optional | SQLite path used by the MCP server and the client SDK; falls back to `ATAGIA_SQLITE_PATH` in the client. |
| `ATAGIA_STORAGE_BACKEND` | `inprocess` | Optional | Storage backend selector. `inprocess` keeps streams in memory; `redis` uses Redis Streams. |
| `ATAGIA_REDIS_URL` | `redis://localhost:6379/0` | Optional | Redis connection URL when `ATAGIA_STORAGE_BACKEND=redis`. |
| `ATAGIA_MIGRATIONS_PATH` | `./migrations` (or packaged) | Optional | Directory containing numbered SQL migration files. |
| `ATAGIA_MANIFESTS_PATH` | `./manifests` (or packaged) | Optional | Directory containing assistant mode manifest JSON files. |
| `ATAGIA_OPERATIONAL_PROFILES_PATH` | `./operational_profiles` (or packaged) | Optional | Directory containing operational profile JSON files. |
| `ATAGIA_ARTIFACT_BLOB_STORAGE_KIND` | `sqlite_blob` | Optional | Where artifact blobs are stored. One of `sqlite_blob` or `local_file`. |
| `ATAGIA_ARTIFACT_BLOB_STORAGE_PATH` | `./data/artifact_blobs` | Optional | Directory for artifact blobs when `ATAGIA_ARTIFACT_BLOB_STORAGE_KIND=local_file`. |

---

## 3. LLM provider keys and base URLs

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_ANTHROPIC_API_KEY` | _(unset)_ | Conditional | API key for Anthropic. Required when any resolved component uses an `anthropic/...` model. |
| `ATAGIA_OPENAI_API_KEY` | _(unset)_ | Conditional | API key for OpenAI. Required when any component uses an `openai/...` model or for OpenAI-hosted embeddings. |
| `ATAGIA_GOOGLE_API_KEY` | _(unset)_ | Conditional | API key for Google Gemini. Required when any component uses a `google/...` model. |
| `ATAGIA_OPENROUTER_API_KEY` | _(unset)_ | Conditional | API key for OpenRouter. Required when any component uses an `openrouter/...` model. |
| `ATAGIA_ANTHROPIC_BASE_URL` | _(unset)_ | Optional | Override Anthropic API base URL. |
| `ATAGIA_OPENAI_BASE_URL` | _(unset)_ | Optional | Override OpenAI API base URL. |
| `ATAGIA_OPENAI_EMBEDDING_BASE_URL` | _(unset)_ | Optional | Override only the OpenAI-compatible embeddings base URL. When unset, embeddings use `ATAGIA_OPENAI_BASE_URL`. |
| `ATAGIA_OPENROUTER_BASE_URL` | _(unset)_ | Optional | Override OpenRouter API base URL. |
| `ATAGIA_OPENROUTER_SITE_URL` | `http://localhost` | Optional | `HTTP-Referer` header sent to OpenRouter for attribution. |
| `ATAGIA_OPENROUTER_APP_NAME` | `Atagia` | Optional | `X-Title` header sent to OpenRouter for attribution. |

## 4. LLM model selection

Model specs are provider-qualified: `provider/model[,thinking_level]`, e.g.
`anthropic/claude-sonnet-4-6` or `openrouter/google/gemini-3.1-flash-lite`.
Resolution order per component: forced-global -> component override ->
category override -> built-in default.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_LLM_FORCED_GLOBAL_MODEL` | _(unset)_ | Optional | Force a single model for every completion component (highest priority). |
| `ATAGIA_LLM_INGEST_MODEL` | _(unset)_ | Optional | Category override for all ingest-side components. |
| `ATAGIA_LLM_RETRIEVAL_MODEL` | _(unset)_ | Optional | Category override for all retrieval-side components. |
| `ATAGIA_LLM_CHAT_MODEL` | _(unset)_ | Optional | Category override for the chat component. |
| `ATAGIA_LLM_MODEL__<COMPONENT_ID>` | _(unset)_ | Optional | Per-component override. `<COMPONENT_ID>` is one of the IDs listed below, uppercased. |

### Component IDs (for `ATAGIA_LLM_MODEL__<COMPONENT_ID>`)

Ingest: `EXTRACTOR`, `TEXT_CHUNKER`, `COMPACTOR`, `SUMMARY_PRIVACY_JUDGE`,
`SUMMARY_PRIVACY_REFINER`, `BELIEF_REVISER`, `CONTRACT_PROJECTION`,
`GRAPH_PROJECTION`, `CONSEQUENCE_BUILDER`, `CONSEQUENCE_DETECTOR`,
`TOPIC_WORKING_SET`, `CONSENT_CONFIRMATION`, `INTENT_CLASSIFIER`,
`EXTRACTION_WATCHDOG`, `EXPORT_ANONYMIZER`.

Retrieval: `NEED_DETECTOR`, `COVERAGE_EXPANDER`, `APPLICABILITY_SCORER`,
`CONTEXT_STALENESS`, `METRICS_COMPUTER`.

Chat: `CHAT`.

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

### Answer context envelope

Answer-time prompts use one structural input envelope by default. The envelope
allocates the full global budget across instructions, the current turn,
retrieved context, and recent transcript; it does not force empty filler when a
section has less useful material than its allocation. The current default is
the retained-replay calibrated 4k budget.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_CONTEXT_ENVELOPE_BUDGET_TOKENS` | `4096` | Optional | Global answer-input envelope budget used to derive section budgets for retrieved context and recent transcript. |
| `ATAGIA_CONTEXT_ENVELOPE_RATIOS` | `instructions=0.10,current_turn=0.03,retrieved_context=0.67,recent_transcript=0.20` | Optional | Section allocation ratios. Accepts either a JSON object or comma-separated `key=value` pairs; values are normalized before allocation. |

---

## 5. Embeddings

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_EMBEDDING_BACKEND` | `none` | Optional | Embedding backend. `none` runs FTS-only retrieval; `sqlite_vec` enables hybrid vector search. |
| `ATAGIA_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Optional | Provider-qualified embedding model spec. |
| `ATAGIA_EMBEDDING_DIMENSION` | `1536` | Optional | Embedding vector dimension. Must match the configured embedding model's output dimension. |
| `ATAGIA_EMBEDDING_VECTOR_LIMIT_CAP` | `50` | Optional | Maximum vector candidates returned by sqlite-vec per query before applicability scoring. |
| `ATAGIA_EMBEDDING_SEARCH_OVERFETCH_MULTIPLIER` | `4` | Optional | Multiplier applied to the requested limit when over-fetching vector candidates before fusion. |

---

## 6. Intimacy fallback policy

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

## 7. Chunking

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_DISABLE_CHUNKING_EXTRACTION` | `false` | Optional | When true, bypass two-level chunking during extraction and pass full text directly to the extractor. |
| `ATAGIA_CHUNKING_EXTRACTION_THRESHOLD_TOKENS` | `2048` | Optional | Token threshold above which the extractor activates chunked extraction. Must be positive. |

---

## 8. Initial context package rollout

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

## 9. Worker circuit breaker

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

## 10. Debug and observability

These are intended for local development and benchmark diagnostics. Keep
disabled in production. When enabled, raw prompt/response artifacts are
written to disk under `ATAGIA_DEBUG_LLM_IO_DIR`, one JSON file per LLM call
with optional raw request/response bodies.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_DEBUG` | `false` | Optional | Generic debug toggle for verbose logging. |
| `ATAGIA_DEBUG_LLM_IO` | `false` | Optional | Persist every LLM call to disk for inspection. |
| `ATAGIA_DEBUG_LLM_IO_DIR` | `./docs/tmp/llm_debug` | Optional | Directory where LLM IO artifacts are written. |
| `ATAGIA_DEBUG_LLM_IO_PURPOSES` | _(empty)_ | Optional | Comma-separated allowlist of purposes to record. Empty means record all. |
| `ATAGIA_DEBUG_LLM_IO_RAW` | `false` | Optional | Also persist raw provider request/response payloads alongside the structured artifact. |
| `ATAGIA_DEBUG_LLM_IO_MAX_CHARS` | `50000` | Optional | Maximum characters per recorded field before truncation. |

---

## 11. Service mode

Service mode runs Atagia as a FastAPI HTTP service. The library mode trusts
the caller's `user_id`; the service mode requires an API key.

| Variable | Default | Required | Description |
|---|---|---|---|
| `ATAGIA_SERVICE_MODE` | `false` | Optional | Enable HTTP service mode. The shipped `.env.example` sets this to `true`. |
| `ATAGIA_SERVICE_API_KEY` | _(unset)_ | Required for service mode | API key required by every non-admin HTTP endpoint. |
| `ATAGIA_ADMIN_API_KEY` | _(unset)_ | Required for admin endpoints | API key required by admin endpoints. Used by the client SDK for admin operations. |
| `ATAGIA_BASE_URL` | _(unset)_ | Optional | Base URL used by the client SDK to reach the service. |
| `ATAGIA_ALLOW_INSECURE_HTTP` | `false` | Optional | Local-only escape hatch that allows non-TLS HTTP outside loopback. Keep `false` in production. |
| `ATAGIA_CORS_ALLOWED_ORIGINS` | _(empty)_ | Optional | Comma-separated origin allowlist for browser-based clients (e.g. `http://127.0.0.1:8000`). |
| `ATAGIA_WORKERS_ENABLED` | `false` | Optional | Start background workers alongside the HTTP service. The shipped `.env.example` sets this to `true`. |
| `ATAGIA_PROXY_MODEL_ID` | `atagia-memory-proxy` | Optional | Visible model id surfaced to OpenAI-compatible proxy clients. |
| `ATAGIA_PROXY_UPSTREAM_MODEL` | _(unset)_ | Optional | Provider-qualified upstream model used by the proxy. Falls back to the configured chat model when unset. |
| `ATAGIA_PROXY_DEFAULT_MODE` | _(unset)_ | Optional | Default assistant mode applied by the proxy when the client does not send one. |

---

## 12. MCP server

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

## 13. Sidecar bridge and client facade

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

## 14. Example environments

### 14.1. Minimal single-provider smoke env

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

### 14.2. Mixed-provider production env with intimacy fallback

Default routing (OpenRouter Gemini Flash-Lite for ingest/retrieval, OpenRouter
DeepSeek v4 Flash for ordinary chat, Anthropic for privacy/consent/export
components, and direct Anthropic Opus 4.7 for benchmark judging) plus OpenAI
embeddings, service mode behind an API key, and a per-component intimacy
fallback for the extractor and compactor.

```env
ATAGIA_ANTHROPIC_API_KEY=your-anthropic-key
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

### 14.3. Local Ollama runtime benchmark env

Ollama's OpenAI-compatible endpoint can be used through the `openai/...`
provider namespace. This is a convenient starting point for local
retrieval/answer experiments without a paid API.

```env
ATAGIA_OPENAI_API_KEY=ollama
ATAGIA_OPENAI_BASE_URL=http://localhost:11434/v1
ATAGIA_OPENAI_EMBEDDING_BASE_URL=http://localhost:11434/v1
ATAGIA_LLM_FORCED_GLOBAL_MODEL=openai/qwen3-coder:30b
ATAGIA_EMBEDDING_BACKEND=sqlite_vec
ATAGIA_EMBEDDING_MODEL=openai/qwen3-embedding:4b
ATAGIA_EMBEDDING_DIMENSION=1536
```

Local model speed and quality vary widely by hardware and quantization;
measure on your own setup before drawing conclusions.
