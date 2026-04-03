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
- **Adaptive context caching** -- deterministic staleness scoring serves cached results on follow-up turns when context has not significantly changed
- **Intelligent text chunking** -- two-level splitting (rule-based + AI-assisted) handles voice transcriptions and long pastes before extraction
- **All storage in SQLite** -- no external vector DB required; sqlite-vec available for optional embedding recall

## Current status

Atagia is functional and under active development. The core pipeline (extraction, retrieval, scoring, belief revision, consequence chains) is implemented and tested with 400+ unit and integration tests.

What works today:
- Full memory extraction from conversations with LLM-based applicability scoring
- Belief revision with 8 conflict resolution strategies and full version history
- Consequence chain learning (action to outcome to tendency)
- Interaction contract learning (how the user prefers to collaborate)
- Adaptive context cache with deterministic staleness scoring
- Intelligent two-level text chunking for long messages (voice transcriptions, etc.)
- Message timestamps for temporal reference resolution
- Library mode, REST API, and MCP server
- LoCoMo benchmark harness with ablation support

What is in progress or planned:
- **Retrieval recall improvements**: candidate search currently uses FTS5 (full-text search). Single-hop and temporal questions work well, but multi-hop questions that require finding specific facts across hundreds of memories need better candidate generation. Vector embeddings (sqlite-vec) are the planned next step.
- **Full benchmark validation**: the LoCoMo harness runs end-to-end but published accuracy numbers are pending the embedding work above.
- **Neo4j graph layer**: planned for cases where relationship traversal adds value over flat retrieval. Will only ship if benchmark evidence justifies the added complexity.

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
| `llm_provider` | From env | `"anthropic"`, `"openai"`, or `"openrouter"` |
| `llm_api_key` | From env | API key for the LLM provider |
| `llm_model` | From env | Model name for extraction, scoring, and chat |
| `embedding_backend` | `"none"` | `"none"` or `"sqlite_vec"` |
| `embedding_model` | `None` | Embedding model name (required when backend is `sqlite_vec`) |
| `context_cache_enabled` | `True` | Enable adaptive context caching |
| `chunking_enabled` | `True` | Enable intelligent chunking for long messages |
| `skip_belief_revision` | `False` | Disable belief revision (for benchmarks/ablation) |
| `skip_compaction` | `False` | Disable compaction (for benchmarks/ablation) |

SQLite is the only required storage dependency. An LLM API (Anthropic, OpenAI, or OpenRouter) is required for memory extraction, scoring, and chat. Redis accelerates queues and caching but is optional -- the engine works without it using in-process queues.

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
| POST | `/v1/conversations` | Create a conversation |
| POST | `/v1/workspaces` | Create a workspace |
| POST | `/v1/chat/{conversation_id}/reply` | Send a message and get a response |
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

Cache entries are bound to the active policy hash (manifest changes force misses), validated against the current workspace, and invalidated on any mutation (new messages, memory edits, rebuilds, workspace changes). MCP and benchmark paths always use fresh retrieval.

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

## Benchmarking

Atagia includes a LoCoMo benchmark harness with a dataset downloader, LLM judge scorer, ablation presets, and CLI summary output.

```bash
python -m benchmarks.locomo.download
python -m benchmarks.locomo \
  --data-path benchmarks/data/locomo10.json \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --max-questions 25
```

Available ablation presets: `similarity_only`, `no_contract`, `no_scope`, `no_need_detection`, `no_revision`, `no_compaction`.

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Extraction, retrieval, scoring, contracts, lifecycle, API, workers | Done |
| 2 | Belief revision, consequence chains, compaction, evaluation, replay | Done |
| 2.5 | Library mode, MCP server, documentation | Done |
| 3 | LoCoMo benchmark harness, FTS multi-query retrieval, ablation support | Done |
| 3.5 | Adaptive context cache, staleness scoring, invalidation hardening | Done |
| 3.6 | Intelligent text chunking for long messages | Done |
| 3.7 | Message timestamps and temporal reference resolution | Done |
| 4 | Vector embeddings (sqlite-vec) for improved candidate recall | Next |
| 5 | Neo4j graph layer for relationship-aware retrieval | Planned |

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
