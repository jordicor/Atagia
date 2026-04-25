# Atagia

Open-source memory engine for AI assistants. Selects memories by **applicability**, not similarity.

Named after *autophagy*, the cellular process of recycling what no longer serves.

## Why

AI assistants forget everything between sessions. The standard fix -- retrieve memories by embedding similarity -- creates its own problems: coding preferences bleed into emotional support conversations, outdated beliefs override current context, and compressed summaries get treated as established facts. Similarity tells you what *sounds related*, not what *actually helps*.

Atagia scores each candidate memory across multiple dimensions (task fit, mode fit, temporal validity, epistemic quality, risk relevance) and selects based on **applicability** to the current situation. The result is memory that adapts to what the assistant is doing right now, not just what the user said before.

## Key capabilities

- **Applicability-based memory selection** -- memories are scored on whether they help the current task, not just whether they sound similar
- **Belief revision with version history** -- 8 conflict resolution strategies; beliefs are never silently overwritten
- **Scoped personalization per assistant mode** -- a coding assistant and a companion retrieve different memories from the same user, governed by policy manifests
- **Consequence chain learning** -- records action-outcome-tendency chains so the assistant can learn from prior advice results
- **Interaction contract learning** -- observes how the user prefers to collaborate (depth, directness, pushback tolerance) and adapts per mode
- **Natural memory capture** -- picks up facts from normal conversation without requiring "remember this" commands
- **Consent-gated memory** -- sensitive information is stored only after user confirmation, with per-user category tracking
- **Temporal grounding** -- resolves relative dates ("last Saturday", "three weeks ago") against source timestamps into actual calendar dates
- **Adaptive context caching** -- deterministic staleness scoring serves cached results on follow-up turns when context has not significantly changed
- **Operational profiles (experimental)** -- per-request runtime condition presets (`normal`, `low_power`, `offline`, `emergency`, `disaster`) are normalized, authorized, and carried through cache/events/jobs
- **Two-level text chunking** -- rule-based + AI-assisted splitting handles voice transcriptions and long pastes before extraction
- **All storage in SQLite** -- no external vector DB required; sqlite-vec available for optional embedding recall

## Current status

Atagia is functional and under active development. The core pipeline is implemented and covered by a broad unit, integration, API, MCP, worker, and benchmark-oriented test suite.

What works today:
- Memory extraction from conversations with LLM-based applicability scoring
- Natural memory capture from casual conversation (no protocol phrases required)
- Consent-gated memory storage with per-user category thresholds
- Hybrid retrieval: FTS5 with reciprocal rank fusion and progressive multi-query expansion
- Three-level memory hierarchy (L0 verbatim, L1 belief, L2 summary) with mirror retrieval
- Temporal grounding for relative dates against source message timestamps
- Belief revision with 8 strategies and full version history
- Consequence chain learning and interaction contract observation
- Adaptive context cache with deterministic staleness scoring
- Operational profile plumbing for runtime condition-aware integrations
- Optional two-level text chunking for long messages (voice transcriptions, pastes)
- Query-aware context selection with diversity reranking
- Library mode, REST API, and MCP server
- LoCoMo benchmark harness with ablation support and replay probes

What is in progress or planned:
- **Vector embeddings**: candidate search now supports optional sqlite-vec semantic recall on top of FTS5 for local/pre-alpha recall experiments. Service mode rejects sqlite-vec until vector ranking is user-partitioned before ranking.
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
    llm_provider="anthropic",      # or "openai", "openrouter"
    llm_api_key="sk-ant-...",
) as engine:
    # Create resources
    await engine.create_user("user_1")
    await engine.create_conversation("user_1", "conv_1", assistant_mode_id="coding_debug")

    # Get memory-enriched context for your own LLM call
    context = await engine.get_context(
        user_id="user_1",
        conversation_id="conv_1",
        message="What did we decide about the migration?",
        mode="coding_debug",
    )
    # context.system_prompt       -> inject into your LLM
    # context.memories            -> scored memories that were selected
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

Same-process local mode:

```python
from atagia.client import connect_atagia

client = await connect_atagia(
    transport="local",
    db_path="memory.db",
    llm_provider="anthropic",
    llm_api_key="sk-ant-...",
)

context = await client.get_context(
    user_id="user_1",
    conversation_id="conv_1",
    message="What did we decide about the migration?",
    mode="coding_debug",
)

response_text = await my_llm_call(
    system_prompt=context.system_prompt,
    user_text="What did we decide about the migration?",
)

await client.add_response(
    user_id="user_1",
    conversation_id="conv_1",
    text=response_text,
)
await client.close()
```

HTTP service mode:

```python
from atagia.client import connect_atagia

client = await connect_atagia(
    transport="http",
    base_url="http://localhost:8100",
    api_key="your-service-api-key",
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
    await engine.create_conversation("user_1", "conv_1")

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
| `manifests_dir` | Built-in | Directory with assistant mode JSON manifests |
| `operational_profiles_dir` | Built-in | Directory with canonical operational profile JSON presets |
| `llm_provider` | From env | `"anthropic"`, `"openai"`, or `"openrouter"` |
| `llm_api_key` | From env | API key for the LLM provider |
| `llm_model` | From env | Model name for extraction, scoring, and chat |
| `embedding_backend` | `"none"` | `"none"` or `"sqlite_vec"` |
| `embedding_provider_name` | From env | Optional override for which provider handles embeddings |
| `embedding_model` | `None` | Embedding model name (required when backend is `sqlite_vec`) |
| `context_cache_enabled` | `True` | Enable adaptive context caching |
| `chunking_enabled` | From env (`False` by default) | Enable intelligent chunking for long messages |
| `skip_belief_revision` | `False` | Disable belief revision (for benchmarks/ablation) |
| `skip_compaction` | `False` | Disable compaction (for benchmarks/ablation) |

SQLite is the only required storage dependency. An LLM API (Anthropic, OpenAI, or OpenRouter) is required for memory extraction, scoring, and chat. Redis accelerates queues and caching but is optional -- the engine works without it using in-process queues.

For a dual-provider setup, a common production shape is Anthropic for chat/extraction/scoring plus sqlite-vec embeddings on OpenAI or OpenRouter. Example env:

```bash
ATAGIA_LLM_PROVIDER=anthropic
ATAGIA_LLM_API_KEY=your-anthropic-key
ATAGIA_OPENAI_API_KEY=your-openai-key
ATAGIA_EMBEDDING_BACKEND=sqlite_vec
ATAGIA_EMBEDDING_PROVIDER=openai
ATAGIA_EMBEDDING_MODEL=text-embedding-3-small
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
        "ATAGIA_LLM_PROVIDER": "anthropic",
        "ATAGIA_LLM_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Five tools are exposed: `atagia_get_context`, `atagia_add_memory`, `atagia_search_memories`, `atagia_list_memories`, `atagia_delete_memory`.

### As a REST API

```bash
git clone https://github.com/jordicor/atagia.git
cd atagia
pip install -e ".[dev]"
cp .env.example .env   # configure LLM provider and keys
uvicorn atagia.app:create_app --factory --reload
```

Service mode requires `ATAGIA_SERVICE_MODE=true` and `ATAGIA_SERVICE_API_KEY` in `.env`.

#### Core routes

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/users` | Create a user |
| POST | `/v1/conversations` | Create a conversation |
| POST | `/v1/workspaces` | Create a workspace |
| POST | `/v1/chat/{conversation_id}/reply` | Send a message and get a response |
| POST | `/v1/conversations/{conversation_id}/context` | Get sidecar context for a host-managed LLM call |
| POST | `/v1/conversations/{conversation_id}/responses` | Persist a host-generated assistant response |
| POST | `/v1/conversations/{conversation_id}/messages` | Ingest a user or assistant message without retrieval |
| POST | `/v1/flush` | Wait for pending background work |
| POST | `/v1/memory/feedback` | Submit memory feedback (used, useful, irrelevant, intrusive, stale) |
| GET | `/v1/memory/objects/{memory_id}` | Inspect a memory object |
| GET | `/v1/users/{user_id}/contract` | View the user's interaction contract |
| GET | `/v1/users/{user_id}/state` | View the user's current state |

#### Admin routes (require `ATAGIA_ADMIN_API_KEY`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/admin/rebuild/conversation/{id}` | Rebuild memories for a conversation |
| POST | `/v1/admin/rebuild/user/{id}` | Rebuild all memories for a user |
| POST | `/v1/admin/compact/conversation/{id}` | Compact a conversation |
| POST | `/v1/admin/compact/workspace/{id}` | Compact a workspace |
| POST | `/v1/admin/reindex` | Rebuild FTS indexes |
| POST | `/v1/admin/lifecycle/run` | Run memory lifecycle (decay, archival) |
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

### Scoped personalization

Memory is not a flat global profile. Each assistant mode defines its own retrieval policy:

- **coding_debug**: prefers evidence, tight scope, low personalization
- **research_deep_dive**: broad scope, high depth, tolerates uncertainty
- **companion**: high emotional sensitivity, prefers interaction contracts
- **brainstorm**: wide association, loose scope filtering
- **biographical_interview**: maximizes evidence recall, strict privacy
- **general_qa**: balanced defaults

Policies control which scopes are allowed, what memory types are preferred, privacy ceilings, context budgets, and retrieval parameters. Custom modes are defined as JSON manifests.

### Operational profiles

Operational profiles describe the runtime conditions for one request. Assistant modes answer "what kind of interaction is this?", while operational profiles answer "what condition is the device/person/environment in right now?"

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

Cache entries are bound to the active policy hash and operational profile token (manifest/profile changes force misses), validated against the current workspace, and invalidated on any mutation (new messages, memory edits, rebuilds, workspace changes). MCP and benchmark paths always use fresh retrieval.

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
| LLM providers | Anthropic, OpenAI, OpenRouter |
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

LoCoMo is useful as a community benchmark, but full runs are slow and some ground-truth issues in the dataset are still being audited ([details](https://github.com/snap-research/locomo/issues/35)). Atagia-bench is faster and focuses on the behaviors Atagia is designed to get right: consent gating, privacy and mode boundaries, abstention, belief revision, supersession, exact recall, cross-conversation aggregation, preferences, and multilingual smoke cases.

Current internal signal: 60/64 pass rate (93.8%), with remaining failures concentrated around high-risk secret/code recall policy. These numbers are development signals, not public competitive claims; reproducible public baselines and holdout evaluation are still being hardened.

## Benchmarking

Atagia includes a LoCoMo benchmark harness with a dataset downloader, LLM judge scorer, correction overlays, ablation presets, diff reporting, and CLI summary output.

```bash
python -m benchmarks.locomo.download
python -m benchmarks.locomo \
  --data-path benchmarks/data/locomo10.json \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --max-questions 25
```

Available ablation presets: `similarity_only`, `no_contract`, `no_scope`, `no_need_detection`, `no_revision`, `no_compaction`.

Benchmark status is still pre-alpha. Use the harness for regression tracking and reproducibility work; do not treat current local numbers as a published competitive claim.

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core memory system: extraction, retrieval, scoring, contracts, lifecycle, API | Done |
| 2 | Belief revision, consequence chains, compaction, evaluation, replay | Done |
| 2.5 | Library mode, MCP server | Done |
| 3 | Benchmark foundation, offline ingestion, LoCoMo harness, ablations, reporting | In progress |
| 3.5 | Retrieval quality waves, adaptive context cache, text chunking, temporal grounding, memory hierarchy, natural memory capture, consent gating | Mostly done / stabilizing |
| 4 | Evidence and reliability: reproducible benchmark baselines, high-risk policy, retrieval hardening, compaction validation, relational memory slice | Next |
| 5 | Graph or SaaS expansion | Deferred until benchmark evidence or user demand justifies it |

## Research

Atagia is backed by cross-domain research into memory systems spanning microbiology, neuroscience, physics, and traditional knowledge frameworks:

- [Beyond Similarity: Applicability-Governed Memory](docs/Beyond_Similarity_Applicability_Governed_Memory.md) -- thesis paper with testable hypotheses and evaluation strategy
- [Beyond Human Memory](docs/BEYOND_HUMAN_MEMORY.md) -- cross-domain exploration from cellular autophagy to Aboriginal Dreamtime
- [Technical Design](docs/Atagia_Technical_Design.md) -- full engineering specification

## License

[Apache 2.0](LICENSE)

## Links

- Website: [atagia.org](https://atagia.org)
- Author: Jordi Cor ([Acerting Art Inc.](https://acertingart.com) / OjoCentauri)
