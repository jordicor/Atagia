# Atagia

Open-source memory engine for AI assistants. Selects memories by **applicability**, not similarity.

Named after *autophagy*, the cellular process of recycling what no longer serves.

## Why

AI assistants forget everything between sessions. The standard fix -- retrieve memories by embedding similarity -- creates its own problems: coding preferences bleed into emotional support conversations, outdated beliefs override current context, and compressed summaries get treated as established facts. Similarity tells you what *sounds related*, not what *actually helps*.

Atagia scores each candidate memory across multiple dimensions (task fit, mode fit, temporal validity, epistemic quality, risk relevance) and selects based on **applicability** to the current situation. The result is memory that adapts to what the assistant is doing right now, not just what the user said before.

## Key capabilities

- **Applicability-based memory selection** -- memories are scored on whether they help the current task, not just whether they sound similar
- **Belief revision with version history** -- 8 conflict resolution strategies; beliefs are never silently overwritten
- **Explicit memory identity** -- retrieval is isolated by user, optional persona, platform, character, chat, and reversible incognito gates before ranking
- **Retrieval profiles** -- `mode` selects policy manifests for budgets, preferred memory types, privacy ceilings, and interaction style without acting as a memory namespace
- **Consequence chain learning** -- records action-outcome-tendency chains so the assistant can learn from prior advice results
- **Interaction contract learning** -- observes how the user prefers to collaborate (depth, directness, pushback tolerance) and adapts per mode
- **Natural memory capture** -- picks up facts from normal conversation without requiring "remember this" commands
- **Consent-gated memory** -- sensitive information is stored only after user confirmation, with per-user category tracking
- **Conversation lifecycle and erasure controls** -- close/archive/delete conversations, run temporary sessions, edit/delete memories, and erase a user's data through explicit destructive confirmations
- **Temporal grounding** -- resolves relative dates ("last Saturday", "three weeks ago") against source timestamps into actual calendar dates
- **Adaptive context caching** -- deterministic staleness scoring serves cached results on follow-up turns when context has not significantly changed
- **Topic Working Set orientation** -- keeps a compact view of active and parked conversation topics ahead of raw recent transcript context
- **Operational profiles (experimental)** -- per-request runtime condition presets (`normal`, `low_power`, `offline`, `emergency`, `disaster`) are normalized, authorized, and carried through cache/events/jobs
- **Two-level text chunking** -- rule-based + AI-assisted splitting handles voice transcriptions and long pastes before extraction
- **All storage in SQLite** -- no external vector DB required; sqlite-vec available for optional embedding recall

## Current status

Atagia is functional and under active development. The core pipeline is implemented and covered by a broad unit, integration, API, MCP, worker, and benchmark-oriented test suite.

What works today:
- Memory extraction from conversations with LLM-based applicability scoring
- Canonical memory scopes: `chat`, `character`, and `user`, with `user_id` filtering before any ranking
- Natural memory capture from casual conversation (no protocol phrases required)
- Consent-gated memory storage with per-user category thresholds
- Hybrid retrieval: FTS5 with reciprocal rank fusion and progressive multi-query expansion
- Three-level memory hierarchy (L0 verbatim, L1 belief, L2 summary) with mirror retrieval
- Immediate working memory: recent same-conversation transcript is added by token budget before retrieved long-term memory
- Topic Working Set: compact active/parked topic context is refreshed asynchronously and placed before recent transcript text in prompts
- Conversation lifecycle: active/closed/archived/pending-deletion status, temporary conversations, idle TTL expiry, purge-on-close, and write-path guards for non-active conversations
- Memory edit/delete and right-to-erasure: evidence memory edits preserve history and clear stale embeddings; hard deletes cascade through summaries, artifacts, pins, logs, caches, jobs, and local-file cleanup queues
- Temporal grounding for relative dates against source message timestamps
- Belief revision with 8 strategies and full version history
- Consequence chain learning and interaction contract observation
- Adaptive context cache with deterministic staleness scoring
- Operational profile plumbing for runtime condition-aware integrations
- Optional two-level text chunking for long messages (voice transcriptions, pastes)
- Query-aware context selection with diversity reranking
- Library mode, REST API, and MCP server
- Optional sqlite-vec semantic recall with user-partitioned vector search before nearest-neighbor ranking
- LoCoMo benchmark harness with ablation support and replay probes

What is in progress or planned:
- **Benchmark coverage**: see the [Evaluation status](#evaluation-status) section below for current signals and methodology.
- **Neo4j graph layer**: planned for relationship traversal where flat retrieval is insufficient. Will ship only if benchmark evidence justifies the complexity.

## Quick start

### As a Python library

```bash
pip install -e .
```

```python
from atagia import Atagia

async with Atagia(
    db_path="memory.db",
    anthropic_api_key="sk-ant-...",
    llm_forced_global_model="anthropic/claude-sonnet-4-6",
    # Optional immediate transcript override; effective minimum is 2048 tokens.
    recent_transcript_budget_tokens=6000,
) as engine:
    # Create resources
    await engine.create_user("user_1")
    await engine.create_conversation(
        "user_1",
        "conv_1",
        platform_id="web",
        character_id="project_backend",
        mode="coding_debug",
    )

    # Get memory-enriched context for your own LLM call
    context = await engine.get_context(
        user_id="user_1",
        conversation_id="conv_1",
        message="What did we decide about the migration?",
        mode="coding_debug",
    )
    # context.system_prompt       -> inject into your LLM
    # context.topic_working_set_block -> compact active-topic orientation before transcript
    # context.recent_transcript   -> immediate prior turns included verbatim by budget
    # context.memories            -> scored memories that were selected
    # context.assistant_guidance  -> optional suggestions for natural follow-up when context is omitted
    # context.contract            -> how this user prefers to collaborate
    # context.detected_needs      -> signals detected in the query
    # context.from_cache          -> whether this was served from cache
    # context.staleness           -> 0.0 (fresh) to 1.0 (stale)

    # Or let Atagia handle the LLM call too
    result = await engine.chat(
        user_id="user_1",
        conversation_id="conv_1",
        message="Why is the test failing?",
        mode="coding_debug",
    )
    print(result.response_text)
```

### As a sidecar client for your own webchat

Use `connect_atagia` when your app should work the same way whether Atagia is imported
in-process or running as a local/remote HTTP service.

For host applications, the canonical runtime helpers live in
`atagia.integrations`. The top-level `integrations/` directory contains
copyable examples and host notes only.

Same-process local mode:

```python
from atagia.client import connect_atagia

client = await connect_atagia(
    transport="local",
    db_path="memory.db",
    anthropic_api_key="sk-ant-...",
    llm_forced_global_model="anthropic/claude-sonnet-4-6",
)

context = await client.get_context(
    user_id="user_1",
    conversation_id="conv_1",
    message="What did we decide about the migration?",
    mode="coding_debug",
    message_id="host:msg:42",
    source_seq=42,
)

response_text = await my_llm_call(
    system_prompt=context.system_prompt,
    user_text="What did we decide about the migration?",
)

await client.add_response(
    user_id="user_1",
    conversation_id="conv_1",
    text=response_text,
    message_id="host:msg:43",
    source_seq=43,
)
await client.close()
```

#### Host application sidecar integration

Use `SidecarBridge` when the host app owns the LLM call and should fail open if
Atagia is disabled or unavailable:

```python
from atagia.integrations import (
    SidecarBridge,
    SidecarBridgeConfig,
    build_injection_decision,
    context_messages_for_provider,
)

bridge = SidecarBridge(
    SidecarBridgeConfig(enabled=True, platform_id="aurvek", mode="personal_assistant")
)

context = await bridge.get_context_for_turn(
    user_id="aurvek:user:7",
    conversation_id="aurvek:conv:91",
    message_text="What did we decide?",
    character_id="prompt:3",
    message_id="aurvek:msg:123",
    source_seq=123,
)
decision = build_injection_decision(existing_system_prompt, context)
context_messages = context_messages_for_provider(context_messages, decision)
```

`message_id` is optional. When supplied to sidecar context, message ingestion, or
assistant-response persistence, retrying the same `message_id` with the same
role and text is idempotent; reusing it for incompatible content returns a
clear conflict.

`source_seq` is optional and intended for historical backfills. When supplied,
Atagia stores it as the conversation order, so retrying an older failed import
preserves chronology instead of appending the message after newer syncs.

Hosts can pause Atagia background processing before maintenance:

```python
await bridge.pause_new_jobs(reason="backup")
await bridge.drain_and_pause(reason="restart", timeout_seconds=30)
await bridge.hard_pause(reason="restore")
await bridge.resume_processing(reason="done")
```

For HTTP mode, these stop-switch calls require `admin_api_key` or
`ATAGIA_ADMIN_API_KEY` because they use `/v1/admin/worker-control`.

Atagia also has an automatic worker circuit breaker. By default, if at least 20
recent worker attempts fail within 180 seconds and those failures are at least
80% of recent attempted work, Atagia switches itself to `hard_pause` with an
admin-visible reason. This protects backfills from repeatedly hitting a failed
provider, a laptop network outage, or bad configuration. Tune with
`ATAGIA_WORKER_CIRCUIT_BREAKER_ENABLED`,
`ATAGIA_WORKER_CIRCUIT_BREAKER_FAILURE_THRESHOLD`,
`ATAGIA_WORKER_CIRCUIT_BREAKER_WINDOW_SECONDS`, and
`ATAGIA_WORKER_CIRCUIT_BREAKER_MIN_FAILURE_RATIO`.

HTTP service mode:

```python
from atagia.client import connect_atagia

client = await connect_atagia(
    transport="http",
    base_url="http://localhost:8100",
    api_key="your-service-api-key",
    admin_api_key="your-admin-api-key",  # only needed for admin controls
)

result = await client.chat(
    user_id="user_1",
    conversation_id="conv_1",
    message="Why is the test failing?",
)
print(result.response_text)
```

Auto mode uses HTTP when `base_url` or `ATAGIA_BASE_URL` is present, otherwise it
uses local mode. It also reads `ATAGIA_SERVICE_API_KEY` for HTTP and `ATAGIA_DB_PATH`
or `ATAGIA_SQLITE_PATH` for local mode:

```python
client = await connect_atagia(transport="auto")
```

MCP remains the right transport for Claude Desktop, Cursor, and other tool clients.
For ordinary backend, desktop app, or webchat integrations, use the Python client
facade instead.

#### Offline ingestion

For loading long conversations without triggering retrieval on every turn:

```python
async with Atagia(db_path="memory.db") as engine:
    await engine.create_user("user_1")
    await engine.create_conversation("user_1", "conv_1", platform_id="web")

    # Ingest messages (extraction runs in background workers)
    await engine.ingest_message("user_1", "conv_1", "user", "I was born in Barcelona in 1990.")
    await engine.ingest_message("user_1", "conv_1", "assistant", "Tell me more about growing up there.")

    # Persist an assistant response in conversation history
    await engine.add_response("user_1", "conv_1", "That sounds like a great place to grow up.")

    # Wait for all background extraction jobs to finish
    await engine.flush(timeout_seconds=60.0)
```

Messages accept an optional `occurred_at` ISO timestamp for historical data where the message happened at a different time than the ingestion. This lets the extraction pipeline resolve temporal references like "yesterday" or "last year" correctly.

#### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `"atagia.db"` | SQLite database path |
| `redis_url` | `None` | Optional Redis URL for cache and queues |
| `manifests_dir` | Built-in | Directory with retrieval profile JSON manifests |
| `operational_profiles_dir` | Built-in | Directory with canonical operational profile JSON presets |
| `anthropic_api_key` / `openai_api_key` / `google_api_key` / `openrouter_api_key` | From env | Provider-specific LLM API keys |
| `llm_forced_global_model` | From env | Optional `provider/model[,thinking_level]` override for all LLM-backed components |
| `llm_intimacy_ingest_model` / `llm_intimacy_retrieval_model` | From env | Optional fallback model for policy-blocked intimacy-sensitive ingest/retrieval calls |
| `llm_intimacy_component_models` | From env | Optional per-component intimacy fallback models |
| `llm_intimacy_proactive_routing_enabled` | `False` | Route already-known intimate contexts directly to configured intimacy models |
| `embedding_backend` | `"none"` | `"none"` or `"sqlite_vec"` |
| `embedding_model` | `openai/text-embedding-3-small` | Provider-qualified embedding model |
| `context_cache_enabled` | `True` | Enable adaptive context caching |
| `disable_chunking_extraction` | From env (`False` by default) | Disable extraction chunking for oversized messages |
| `skip_belief_revision` | `False` | Disable belief revision (for benchmarks/ablation) |
| `skip_compaction` | `False` | Disable compaction (for benchmarks/ablation) |

Extraction chunking is enabled by default and starts above
`ATAGIA_CHUNKING_EXTRACTION_THRESHOLD_TOKENS` (`2048` by default). Use
`ATAGIA_DISABLE_CHUNKING_EXTRACTION=true` only as an operational/debugging
override.

SQLite is the only required storage dependency. LLM models are configured per
component with provider-qualified specs such as `anthropic/claude-sonnet-4-6`
or `openrouter/google/gemini-3.1-flash-lite-preview`. Redis accelerates queues
and caching but is optional -- the engine works without it using in-process
queues.

The built-in defaults use OpenRouter-hosted Gemini Flash-Lite for Tier 1/Tier 2
ingest and retrieval intelligence, while privacy, consent, PII, and chat
components stay on Anthropic Claude Sonnet. The default runtime therefore
requires both `ATAGIA_OPENROUTER_API_KEY` and `ATAGIA_ANTHROPIC_API_KEY`. To run
every component on one provider, set `ATAGIA_LLM_FORCED_GLOBAL_MODEL`.

Intimacy fallback models are optional and inactive by default. They are not used
to proactively route ordinary traffic. If configured, Atagia retries an
LLM-backed component on the fallback model only when the primary provider
returns a policy-block/refusal signal; retry metadata records component id,
purpose, primary model, fallback model, and sanitized error class/reason without
raw conversation text. Category-level fallbacks use:
`ATAGIA_LLM_INTIMACY_INGEST_MODEL` and
`ATAGIA_LLM_INTIMACY_RETRIEVAL_MODEL`. Component overrides use
`ATAGIA_LLM_INTIMACY_MODEL__<COMPONENT_ID>`, for example
`ATAGIA_LLM_INTIMACY_MODEL__EXTRACTOR=openrouter/z-ai/glm-4.6`.
Set `ATAGIA_LLM_INTIMACY_PROACTIVE_ROUTING_ENABLED=true` to skip the primary
model when the request already carries structured intimacy context, such as an
active intimacy policy, a non-ordinary `intimacy_boundary` on source memories or
topics, or explicit host-provided request metadata.

For local benchmark/debug sessions, `ATAGIA_DEBUG_LLM_IO=true` writes opt-in LLM
diagnostic artifacts under `ATAGIA_DEBUG_LLM_IO_DIR` for selected purposes such
as `applicability_scoring`. Set `ATAGIA_DEBUG_LLM_IO_RAW=true` only in local
development when raw prompts/responses are acceptable to retain.

Example mixed-provider env:

```bash
ATAGIA_ANTHROPIC_API_KEY=your-anthropic-key
ATAGIA_OPENROUTER_API_KEY=your-openrouter-key
ATAGIA_OPENAI_API_KEY=your-openai-key
ATAGIA_EMBEDDING_BACKEND=sqlite_vec
ATAGIA_EMBEDDING_MODEL=openai/text-embedding-3-small
ATAGIA_EMBEDDING_DIMENSION=1536
```

Single-provider development env:

```bash
ATAGIA_ANTHROPIC_API_KEY=your-anthropic-key
ATAGIA_LLM_FORCED_GLOBAL_MODEL=anthropic/claude-sonnet-4-6
```

Gemini can be used natively with the Google Gen AI SDK:

```bash
ATAGIA_GOOGLE_API_KEY=your-google-key
ATAGIA_LLM_FORCED_GLOBAL_MODEL=google/gemini-3-flash-preview
ATAGIA_EMBEDDING_MODEL=google/gemini-embedding-2
ATAGIA_EMBEDDING_DIMENSION=1536
```

### As an MCP server (Claude Desktop, Cursor, Windsurf)

```bash
pip install "atagia[mcp]"
```

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "atagia-memory": {
      "command": "/path/to/.venv/bin/atagia-mcp",
      "env": {
        "ATAGIA_DB_PATH": "/path/to/memory.db",
        "ATAGIA_USER_ID": "desktop-user",
        "ATAGIA_PLATFORM_ID": "claude-desktop",
        "ATAGIA_CONVERSATION_ID": "default-desktop-chat",
        "ATAGIA_ANTHROPIC_API_KEY": "sk-ant-...",
        "ATAGIA_LLM_FORCED_GLOBAL_MODEL": "anthropic/claude-sonnet-4-6"
      }
    }
  }
}
```

Five tools are exposed: `atagia_get_context`, `atagia_add_memory`, `atagia_search_memories`, `atagia_list_memories`, `atagia_delete_memory`.

### As a REST API

```bash
git clone https://github.com/jordicor/Atagia.git
cd Atagia
pip install -e ".[dev]"
cp .env.example .env   # configure LLM provider and keys
atagia-api --host 127.0.0.1 --port 8100 --reload
```

Service mode requires `ATAGIA_SERVICE_MODE=true` and `ATAGIA_SERVICE_API_KEY` in `.env`.

#### Core routes

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/users` | Create a user |
| POST | `/v1/conversations` | Create a conversation, including optional temporary-session settings |
| POST | `/v1/workspaces` | Create a workspace |
| POST | `/v1/chat/{conversation_id}/reply` | Send a message and get a response |
| GET | `/v1/models` | OpenAI-compatible model list for memory-proxy clients |
| POST | `/v1/chat/completions` | OpenAI-compatible memory proxy with streaming support |
| POST | `/v1/conversations/{conversation_id}/context` | Get sidecar context for a host-managed LLM call |
| POST | `/v1/conversations/{conversation_id}/responses` | Persist a host-generated assistant response |
| POST | `/v1/conversations/{conversation_id}/messages` | Ingest a user or assistant message without retrieval |
| POST | `/v1/conversations/{conversation_id}/close` | Close a conversation; purge when confirmed |
| POST | `/v1/conversations/{conversation_id}/archive` | Archive a conversation and hide it from default retrieval/listing |
| POST | `/v1/conversations/{conversation_id}/delete` | Hard-delete a conversation and derived data with confirmation |
| POST | `/v1/flush` | Wait for pending background work |
| POST | `/v1/memory/feedback` | Submit memory feedback (used, useful, irrelevant, intrusive, stale) |
| GET | `/v1/memory/objects/{memory_id}` | Inspect a memory object |
| PATCH | `/v1/memories/{memory_id}` | Edit an active evidence memory and preserve edit history |
| POST | `/v1/memories/{memory_id}/delete` | Archive or hard-delete a memory with confirmation |
| GET | `/v1/users/{user_id}/contract` | View the user's interaction contract |
| GET | `/v1/users/{user_id}/state` | View the user's current state |
| POST | `/v1/users/{user_id}/erase` | Right-to-erasure cascade with confirmation |

#### Admin routes (require `ATAGIA_ADMIN_API_KEY`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/admin/worker-control` | Inspect background-processing stop-switch state |
| POST | `/v1/admin/worker-control` | Set `active`, `pause_new_jobs`, `drain_and_pause`, or `hard_pause` |
| POST | `/v1/admin/rebuild/conversation/{id}` | Rebuild memories for a conversation |
| POST | `/v1/admin/rebuild/user/{id}` | Rebuild all memories for a user |
| POST | `/v1/admin/compact/conversation/{id}` | Compact a conversation |
| POST | `/v1/admin/compact/workspace/{id}` | Compact a workspace |
| POST | `/v1/admin/reindex` | Rebuild FTS indexes |
| POST | `/v1/admin/lifecycle/run` | Run memory lifecycle (decay, archival, temporary TTL expiry, tombstone/file cleanup) |
| POST | `/v1/admin/metrics/compute` | Compute retrieval quality metrics |
| GET | `/v1/admin/metrics/latest` | Latest metric values |
| GET | `/v1/admin/metrics/{name}/history` | Metric history over time |
| GET | `/v1/admin/metrics/retrieval-summary` | Retrieval performance summary |
| GET | `/v1/admin/retrieval-events/{id}` | Inspect a retrieval event |
| GET | `/v1/admin/consequence-chains/{user_id}` | List consequence chains for a user |
| POST | `/v1/admin/replay/event/{id}` | Replay a retrieval event with ablation |
| POST | `/v1/admin/replay/conversation/{id}` | Replay a full conversation |
| POST | `/v1/admin/grounding/{event_id}` | Analyze grounding for a retrieval event |
| POST | `/v1/admin/export/conversation/{id}` | Export a conversation |

## How it works

### Four memory layers

| Layer | What it stores | How it updates |
|---|---|---|
| **Evidence** | Verbatim spans, extracted events, citations, timestamps | Append-only. What actually happened. |
| **Belief** | Revisable interpretations derived from evidence | Versioned. Never silently overwritten. |
| **Interaction contract** | How the user prefers to collaborate: depth, directness, pushback tolerance, pace | Learned from observation. Scoped per mode. |
| **State** | Current context: urgency, focus, frustration | Continuously updated. Transient. |

### Applicability scoring

Each candidate memory is scored across multiple dimensions:

```
final_score = 0.65 * llm_applicability
            + 0.15 * retrieval_score
            + 0.10 * vitality_boost
            + 0.10 * confirmation_boost
            - privacy_penalty
            - contradiction_penalty
```

The LLM evaluates applicability by considering task fit, mode fit, temporal validity, epistemic quality, and risk relevance. Semantic similarity contributes to candidate generation but does not govern final selection.

### Memory identity and retrieval profiles

Memory is not a flat global profile. Every retrieval starts with `user_id`, then
applies persona, platform, character, conversation, incognito, sensitivity,
privacy, status, and policy gates before ranking. The canonical memory scopes are:

- **`chat`**: same concrete `conversation_id`
- **`character`**: cross-chat continuity for the same `character_id`
- **`user`**: user-level continuity, still gated by persona, platform, and policy

`mode` selects a retrieval profile. It tunes how memory is used, but it is not a
memory namespace:

- **coding_debug**: prefers evidence, tight scope, low personalization
- **research_deep_dive**: broad scope, high depth, tolerates uncertainty
- **companion**: high emotional sensitivity, prefers interaction contracts
- **brainstorm**: wide association, loose scope filtering
- **biographical_interview**: maximizes evidence recall, strict privacy
- **personal_assistant**: cross-chat continuity, contracts and state first
- **general_qa**: balanced defaults

Profiles control what memory types are preferred, privacy ceilings, context
budgets, retrieval parameters, need triggers, and contract priorities. Custom
profiles are defined as JSON manifests. Legacy code and database columns may
still use names like `assistant_mode_id` or `workspace_id`; public integrations
should prefer `mode` and `character_id`.

### Operational profiles

Operational profiles describe the runtime conditions for one request. Retrieval profiles answer "what kind of interaction is this?", while operational profiles answer "what condition is the device/person/environment in right now?"

Phase 0 is intentionally behavior-neutral: Atagia validates and authorizes profiles, includes them in context-cache keys, stores profile snapshots in cache entries and job envelopes, and logs them in retrieval events. The built-in profile policy overrides are empty, so existing clients that do not pass `operational_profile` behave exactly like `operational_profile="normal"`.

Canonical profiles are sealed for now: `normal`, `low_power`, `offline`, `emergency`, and `disaster`. High-risk profiles are opt-in via settings before they can be used.

### Belief revision

When new evidence conflicts with an existing belief, the system chooses among eight actions:

`REINFORCE` | `WEAKEN` | `SUPERSEDE` | `SPLIT_BY_MODE` | `SPLIT_BY_SCOPE` | `SPLIT_BY_TIME` | `MARK_EXCEPTION` | `ARCHIVE`

Every revision preserves the full history. A belief like "user prefers detailed answers" does not get silently replaced. Instead, it becomes "depth preference is mode-dependent: concise for debugging, deep for research."

### Consequence chains

When a user reports an outcome of prior advice, the system records the chain:

```
action: "Suggested large refactor"
  -> outcome: "User came back with regressions"
  -> tendency: "This workspace is fragile to sweeping changes"
```

These chains surface during retrieval when follow-up failure or loop signals are detected.

### Adaptive context cache

Atagia caches retrieval results and serves them on follow-up turns when the context has not significantly changed. A deterministic staleness scorer evaluates message count, elapsed time, topic continuity, and interaction pace to decide whether to serve from cache or refresh.

Cache entries are bound to the active policy hash and operational profile token
(manifest/profile changes force misses), validated against the current memory
identity, and invalidated on mutations such as new messages, memory edits,
rebuilds, or character/project changes. MCP and benchmark paths always use fresh
retrieval.

### Intelligent chunking

Long messages (voice transcriptions, copy-pasted conversations) are automatically split before memory extraction:

- **Level 0** (rule-based, zero cost): splits at natural boundaries like timestamps, speaker turns, section breaks, and bracketed annotations. Segments below 500 chars are merged with neighbors.
- **Level 1** (AI-assisted): for segments still exceeding 16K tokens after Level 0, inline markers are inserted every ~1,000 tokens and an LLM identifies semantic cut points. Falls back to deterministic size-bounded splitting if the AI output is unusable, with full logging of the fallback event.

Each chunk is extracted independently with a cross-chunk context accumulator that prevents duplicate extraction and helps resolve coreferences across chunks.

## Architecture

```
               Library mode                    Service mode
            +----------------+              +------------------+
            |  Your app      |              |  Any HTTP client  |
            |  imports       |              |  calls            |
            |  Atagia()      |              |  REST API         |
            +-------+--------+              +--------+---------+
                    |                                |
                    +------------ + ----------------+
                                  |
                     +------------v-----------+
                     |   Context Cache        |
                     |   Staleness scoring    |
                     +------------+-----------+
                                  |
                     +------------v-----------+
                     |   RetrievalPipeline    |
                     |   Need detection       |
                     |   FTS5 candidate search|
                     |   Applicability score  |
                     |   Context compose      |
                     +------------+-----------+
                                  |
              +-------------------+-------------------+
              |                   |                   |
     +--------v------+   +-------v-------+   +-------v------+
     |   SQLite      |   |   Workers     |   |   Redis      |
     |   FTS5        |   |   Extract     |   |   (optional) |
     |   sqlite-vec  |   |   Chunk       |   |   Queues     |
     |   Canonical   |   |   Revise      |   |   Cache      |
     +--------------+    |   Compact     |   +--------------+
                         |   Evaluate    |
                         +---------------+
```

SQLite is the canonical data store. An LLM API is required for memory extraction, applicability scoring, belief revision, and chat. Redis accelerates queues and caching but is optional.

## Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| API | FastAPI |
| Primary storage | SQLite + FTS5 |
| LLM providers | Anthropic, OpenAI, Google (Gemini), OpenRouter |
| Optional cache/queues | Redis |
| Optional semantic recall | sqlite-vec |

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

For MCP server tests:

```bash
pip install -e ".[dev,mcp]"
python -m pytest tests/ -v
```

## Evaluation status

Atagia is evaluated with both LoCoMo and an internal regression suite, Atagia-bench.

LoCoMo is useful as a community benchmark, but full runs are slow and some ground-truth issues in the dataset are still being audited ([details](https://github.com/snap-research/locomo/issues/35)). Atagia-bench is faster and focuses on the behaviors Atagia is designed to get right: consent gating, privacy and retrieval-profile boundaries, abstention, belief revision, supersession, exact recall, cross-conversation aggregation, preferences, and multilingual smoke cases.

Current internal policy-aligned signal: 63/64 pass rate (98.4%), average score 0.975, with 0 critical errors in the 2026-05-01 all-persona Atagia-bench run. High-risk and privacy-check questions pass at 5/5 under the ordinary-chat policy that withholds raw reusable secret literals. The remaining failure is `rosa-q02`, an answer-policy/grading issue where the current building code evidence is retrieved but the answer model still withholds it as a protected secret. Current numbers remain development signals only until public baselines are frozen.

## Benchmarking

Atagia includes a LoCoMo benchmark harness with a dataset downloader, LLM judge scorer, correction overlays, ablation presets, retained-DB replay, diff reporting, failed-question custody reports, run manifests, and CLI summary output.

```bash
python -m benchmarks.locomo.download
python -m benchmarks.locomo \
  --data-path benchmarks/data/locomo10.json \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --max-questions 25
```

Retained LoCoMo DB snapshots can be listed in text or JSON form:

```bash
python -m benchmarks.locomo --list-benchmark-dbs
python -m benchmarks.locomo --list-benchmark-dbs-json
```

The JSON form includes DB/WAL/SHM byte counts, file mtimes, metadata/progress
hashes, and best-effort SQLite row counts, which is useful for checking
long-running retained-DB ingests.

Active retained-DB runs can also be inspected without waiting for a final report:

```bash
python -m benchmarks.locomo --summarize-run-log docs/tmp/night_run/run.log
python -m benchmarks.locomo --diff-benchmark-db-list before.json after.json
```

Benchmark reports, diffs, custody reports, and run manifests carry warning
counters, selection metadata, retrieval latency and memory-use summaries, and
retrieval-custody summaries for reproducible regression review.

Available ablation presets: `similarity_only`, `no_contract`, `no_scope`, `no_need_detection`, `no_revision`, `no_compaction`.

Benchmark status is still pre-alpha. Use the harness for regression tracking and reproducibility work; do not treat current local numbers as a published competitive claim.

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core memory system: extraction, retrieval, scoring, contracts, lifecycle, API | Done |
| 2 | Belief revision, consequence chains, compaction, evaluation, replay | Done |
| 2.5 | Library mode, MCP server | Done |
| 3 | Benchmark foundation, offline ingestion, LoCoMo harness, ablations, reporting | Done |
| 3.5 | Retrieval quality waves, adaptive context cache, text chunking, temporal grounding, memory hierarchy, natural memory capture, consent gating | Done |
| 4 | Evidence and reliability: reproducible benchmark baselines, high-risk policy, retrieval hardening, compaction validation, relational memory slice | Next |
| 5 | Graph or SaaS expansion | Deferred until benchmark evidence or user demand justifies it |

## Research

Atagia is backed by cross-domain research into memory systems spanning microbiology, neuroscience, physics, and traditional knowledge frameworks:

- [Beyond Similarity: Applicability-Governed Memory](docs/Beyond_Similarity_Applicability_Governed_Memory.md) -- thesis paper with testable hypotheses and evaluation strategy
- [Beyond Human Memory](docs/BEYOND_HUMAN_MEMORY.md) -- cross-domain exploration from cellular autophagy to Aboriginal Dreamtime

## License

[Apache 2.0](LICENSE)

## Links

- Website: [atagia.org](https://atagia.org)
- Author: Jordi Cor ([Acerting Art Inc.](https://acerting.com) / OjoCentauri)
