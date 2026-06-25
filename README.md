# Atagia

*Memory that knows when to forget.*

Atagia is an open-source memory and perspective engine for AI systems that
interact with people across chats, devices, characters, projects, and worlds.
It is not a document-ingestion RAG engine. It is for medium- and long-term
assistant memory: useful continuity without treating every past token as
global, current, or equally relevant.

> "Atagia is memory for AIs, any kind of AI."

An AI rarely lives in one place. The same voice can appear in a chat window, a
domestic robot, an NPC inside a game save, an agent running offline on a
laptop. Each one observes a different slice of the same person, the same
world, the same day. Each has to decide what to remember, what to forget, and
what belongs to a body, a world, or a voice it is not currently inhabiting.
Atagia is the layer underneath that decides — by reading coordinates the host
has already declared, before any ranking happens.

Each candidate memory is scored on whether it actually applies to the
situation in front of it: the task, the active voice, the project, the body,
the world, the moment. Memories that no longer apply are not deleted. They
are recycled — kept as evidence of what once was, retired from current
answers. Atagia is named after autophagy, the cellular process of recycling
what no longer serves.

## Memory coordinates

Memory is not a flat pool. Before a memory is considered for ranking, Atagia
asks where it lives. Whose perspective owns it. Which voice was speaking when
it was captured. Which project, which body, which world it belongs to. These
are coordinates, declared by the host. They gate the candidate pool itself.

| Coordinate | The question it answers | What it makes possible |
|---|---|---|
| **User** | Whose memory is this? | Hard partition. Always first. No exceptions. |
| **Presence** | Which voice was active — assistant, character, facet, or source speaker? | The same AI can run as a focused accountant during the day and a playful companion at night. Memories can be attributed to the voice that lived them instead of silently merging identities. |
| **Space** | Which project, folder, room, or capsule? | A folder can behave like focus mode, a privacy vault, or severance. The host declares the boundary once; retrieval applies it before ranking. |
| **Mind** | Whose internal perspective remembers this? | A user, an NPC, an external actor, and an AI facet can each hold their own version of the same scene without one collapsing into another. The AI knows *who* remembered something, not only *what* was remembered. |
| **Embodiment** | Which body or device captured this? | Capability is bound to the body that has it. A drone's flight envelope does not transfer to a body that cannot fly. What each body sensed about itself stays with it. |
| **Realm** | Which world, reality, simulation, fiction, or game save? | Inside a game, the AI plays the kingdom's advisor. The kingdom's politics stay inside the kingdom. The user's actual job does not appear in the throne room. When asked something the advisor cannot know, it does not pretend to know it. |

An *overseer* topology can read across many local Minds, Spaces, or Realms at
once — but only what has been explicitly granted. Local boundaries remain
intact. The overseer sees what it was given, labeled with where it came from.
Nothing inherits visibility by default.

**Mode** is a retrieval profile, not a coordinate. `coding_debug` prefers
evidence and tight scope. `biographical_interview` maximizes recall with
strict privacy. `companion` leans on interaction contracts. Custom profiles
are JSON manifests.

### Memory topologies

Atagia can be narrow or broad depending on the host and user policy. The same
engine can back a simple assistant, a companion with several prompts, a robot
with several bodies, or a multiplayer world with many local minds.

| Shape | Typical coordinates | Result |
|---|---|---|
| **Ordinary assistant** | `user_id`, `conversation_id`, `mode` | Webchat, desktop assistant, voice assistant, or local agent with cross-chat memory only when the user allows it. |
| **Companion or prompt facets** | `character_id`, Presence, optional `mind_id` | A user can keep characters isolated, let them share attributed memory, or treat them as facets of one wider AI identity. |
| **Project or folder work** | `space_id` with `focus`, `severance`, `privacy_vault`, or `tagged` | A coding project, client folder, research room, or private capsule can focus memory and block leakage across boundaries. |
| **Multi-device or embodied AI** | `platform_id`, `embodiment_id`, operational profile | Phone, laptop, home speaker, robot, drone, or camera memories can keep body-local capabilities and constraints attached to the body that has them. |
| **Games, roleplay, MMORPGs** | `realm_id`, `mind_id`, `mind_topology` | NPCs, player-facing assistants, world advisors, and story entities can keep perspectival memory inside a game save, campaign, simulation, or fictional world. |
| **OjoCentauri / overseer** | `mind_topology=ojocentauri` plus explicit grants | A global authorized view can summarize or coordinate many local Minds, Spaces, or Realms without making local actors omniscient. |

The host can expose these as simple switches: remember across chats, remember
across devices, use an incognito chat, isolate this folder, bridge this Realm,
or grant an overseer a labeled view. Atagia enforces the resulting candidate
pool before ranking, so "more memory" and "less memory" are both first-class
choices rather than prompt wishes.

In practical terms, this means one user can choose a single AI continuity across
many bodies, platforms, and worlds, while another can keep the accountant, the
romantic companion, the campaign NPCs, and the sensitive project folder as
separate memory environments. Both are normal configurations.

## How it works

### Four memory layers

| Layer | What it stores | How it updates |
|---|---|---|
| **Evidence** | Verbatim spans, extracted events, citations, timestamps | Append-only. What actually happened. |
| **Belief** | Revisable interpretations derived from evidence | Versioned. Never silently overwritten. |
| **Interaction contract** | How the user prefers to collaborate: depth, directness, pushback tolerance, pace | Learned from observation. Scoped per mode. |
| **State** | Current context: urgency, focus, frustration | Continuously updated. Transient. |

### Identity and controls

Every retrieval starts with `user_id`. Within that hard partition, Atagia uses
explicit identity and policy fields: `user_persona_id`, `platform_id`,
`character_id`, `conversation_id`, `mode`, `incognito`, Presence, Space, Mind,
Embodiment, and Realm. The canonical storage scopes remain `chat`,
`character`, and `user`; `mode` selects a retrieval profile and is not a memory
namespace.

User and host controls include:

- `remember_across_chats`: whether memory can promote beyond one chat.
- `remember_across_devices`: whether eligible memory can cross platforms.
- `incognito`: reversible chat-only behavior that does not create broad memory.
- `space_id`: project/folder/capsule boundary with focus, severance, vault, or tagged behavior.
- `mind_topology`: one local Mind, many local Minds, or OjoCentauri overseer.
- `embodiment_id` and `realm_id`: body/device and world/domain applicability.

### Applicability scoring

Each candidate memory is scored by an LLM judge against the active situation:
task fit, mode fit, temporal validity, epistemic quality, risk relevance. The
score is combined with lexical and vector retrieval signals, with additional
boosts for memory vitality, prior confirmations, exact recall, and detected
need, and penalties for hallucination risk and temporal staleness. Semantic
similarity contributes to candidate generation but does not govern final
selection.

### Belief revision

When new evidence conflicts with an existing belief, Atagia chooses among
eight actions ranging from reinforcement to scoped split to archival. Every
revision keeps the prior version. A belief like "user prefers detailed
answers" does not get silently replaced. It becomes "depth preference is
mode-dependent: concise for debugging, deep for research."

### Consequence chains

When a user reports the outcome of prior advice, Atagia records the chain:
action → outcome → tendency. These surface during retrieval when follow-up
failure or loop signals are detected.

### Adaptive context cache

Retrieval results are cached and served on follow-up turns when context has
not significantly changed. A deterministic staleness scorer decides; manifest
or operational-profile changes force misses; mutations invalidate.

### Operational profiles

Per-request runtime presets — `normal`, `low_power`, `offline`, `emergency`,
`disaster` — describe the condition of the device or environment for one
request. Atagia validates and authorizes them and carries them through cache
keys and job envelopes. Built-in policy overrides are empty by default;
clients that do not pass `operational_profile` behave exactly like `normal`.
High-risk profiles are opt-in.

### Storage

SQLite is the single source of truth. FTS5 handles lexical retrieval.
sqlite-vec is available as an optional semantic candidate-generation lane.
Redis accelerates queues and caching but is optional.

## Status

**Working today**

- Memory extraction with LLM-based applicability scoring, with `user_id` partitioned first on every query
- Memory coordinates: Presence, Space, Mind, Embodiment, Realm, and canonical `chat` / `character` / `user` scopes
- User/host memory controls for cross-chat memory, cross-device memory, incognito chats, project/folder boundaries, body/device context, and world/domain context
- OjoCentauri overseer topology with grant-mediated Mind/Space/Realm visibility and attribution
- Four memory layers and a three-level hierarchy (verbatim / belief / summary) with mirror retrieval
- Belief revision with version history; consequence chains; interaction contracts
- Hybrid retrieval: FTS5 with reciprocal rank fusion, progressive multi-query expansion, diversity reranking, and an optional sqlite-vec semantic lane
- Adaptive retrieval gate (on by default): turns that only need the model's own knowledge or the visible conversation skip the retrieval stages and answer from the prepared context, with zero added LLM calls; personally anchored or uncertain turns always retrieve
- Adaptive context cache, immediate working memory, and Topic Working Set
- Natural memory capture from ordinary conversation, with consent gating and temporal grounding for relative dates
- Operational profiles (`normal`, `low_power`, `offline`, `emergency`, `disaster`; high-risk opt-in)
- Two-level chunking for long messages
- Conversation lifecycle: active, closed, archived, pending-deletion; plus temporary sessions and idle-TTL expiry
- Memory edit and right-to-erasure cascade
- Coordinate inspector tooling for admin/dev surfaces (memory coordinates, retrieval custody decisions, coordinate-correction audit with cache invalidation)
- Library mode, REST API, MCP server
- LoCoMo benchmark harness with ablation and replay

**In progress**

- Broader coordinate workflows: richer grant UX, end-user review surfaces, audit and rescope flows
- Reproducible public benchmark baselines

**Deferred**

- Graph layer (Neo4j) until benchmark evidence justifies the complexity

## Get started

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
) as engine:
    await engine.create_user("user_1")
    await engine.create_conversation(
        "user_1", "conv_1",
        platform_id="web",
        character_id="project_backend",
        mode="coding_debug",
    )

    context = await engine.get_context(
        user_id="user_1",
        conversation_id="conv_1",
        message="What did we decide about the migration?",
        mode="coding_debug",
    )

    # Or let Atagia handle the LLM call too
    result = await engine.chat(
        user_id="user_1",
        conversation_id="conv_1",
        message="Why is the test failing?",
        mode="coding_debug",
    )
    print(result.response_text)
```

### As a sidecar for your own LLM call

Use `SidecarBridge` when the host application owns generation and Atagia should
only fetch memory context, inject it safely, then store the assistant response.

```python
from atagia.integrations import (
    SidecarBridge,
    SidecarBridgeConfig,
    build_injection_decision,
    context_messages_for_provider,
)

bridge = SidecarBridge(
    SidecarBridgeConfig(
        enabled=True,
        platform_id="aurvek-web",
        mode="personal_assistant",
        space_id="project-atagia",
    )
)

context = await bridge.get_context_for_turn(
    user_id="user_1",
    conversation_id="conv_1",
    message_text="What did we decide?",
    character_id="work-assistant",
    message_id="host-msg-42",
)

decision = build_injection_decision(existing_system_prompt, context)
messages = context_messages_for_provider(existing_messages, decision)
response_text = await my_llm_call(
    system_prompt=decision.system_prompt,
    messages=messages,
    user_text="What did we decide?",
)

await bridge.add_response(
    user_id="user_1",
    conversation_id="conv_1",
    text=response_text,
    message_id="host-msg-43",
)
```

Host applications that own their own LLM call should use `SidecarBridge` from
`atagia.integrations` — see
[docs/HOST_SIDECAR_INTEGRATION.md](docs/HOST_SIDECAR_INTEGRATION.md) for the
fail-open bridge, pause/drain controls, and the worker circuit breaker. For
host apps that want the same code to switch between in-process and HTTP
transports, the `connect_atagia` client facade covers both transports; its
options are documented alongside the bridge.

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

Ten tools are exposed: `atagia_get_context`, `atagia_add_memory`,
`atagia_search_memories`, `atagia_processing_status`, `atagia_list_memories`,
`atagia_edit_memory`, `atagia_delete_memory`, `atagia_close_conversation`,
`atagia_archive_conversation`, `atagia_delete_conversation`.

### As a REST API

```bash
git clone https://github.com/jordicor/Atagia.git
cd Atagia
pip install -e ".[dev]"
cp .env.example .env   # configure LLM provider and keys
atagia-api --host 127.0.0.1 --port 8100 --reload
```

Service mode requires `ATAGIA_SERVICE_MODE=true` and `ATAGIA_SERVICE_API_KEY`.
Core routes cover users, conversations, chat replies, sidecar context, message
ingestion, memory feedback and edits, conversation lifecycle, and erasure. The
full route list including admin endpoints is in
[docs/API.md](docs/API.md).

### Configuration

SQLite is the only required storage dependency. LLM models are configured per
component with provider-qualified specs such as `anthropic/claude-sonnet-4-6`
or `minimax/MiniMax-M3`. Redis is optional.

Default runtime expects `ATAGIA_MINIMAX_API_KEY`,
`ATAGIA_OPENROUTER_API_KEY`, and `ATAGIA_ANTHROPIC_API_KEY`: direct MiniMax M3
handles ingest and compaction intelligence, OpenRouter-hosted Gemini
Flash-Lite handles retrieval intelligence, OpenRouter-hosted DeepSeek v4 Flash
handles cheap ordinary chat answers, and Anthropic Claude Sonnet remains the
default for privacy/consent/export-sensitive components. Benchmark CLIs with
their default judge also expect `ATAGIA_KIMI_API_KEY`. To run every component
on one provider, set `ATAGIA_LLM_FORCED_GLOBAL_MODEL`. Use the stable Gemini
Flash-Lite slug without `-preview` for retrieval overrides. Structured-output
calls perform one same-model corrective retry after JSON/schema validation
fails, with an optional rescue path that escalates only the stuck calls to a
stronger model.
Full configuration — embedding backends, model routing, structured output
repair, intimacy fallback policy, debug LLM I/O — is documented in
[docs/CONFIGURATION_REFERENCE.md](docs/CONFIGURATION_REFERENCE.md).

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

## Evaluation

Atagia is evaluated with LoCoMo as a community benchmark and an internal
regression suite, Atagia-bench, focused on consent gating, privacy boundaries,
abstention, belief revision, exact recall, cross-conversation aggregation,
preferences, and multilingual smoke cases. Full LoCoMo runs are slow, and
some ground-truth issues in the dataset are still being audited.

Current numbers are development signals. Public baselines are not yet frozen.
Use the harness for regression tracking and reproducibility, not as a
competitive claim.

## Research

- [Beyond Similarity: Applicability-Governed Memory](docs/Beyond_Similarity_Applicability_Governed_Memory.md) — thesis paper with testable hypotheses and evaluation strategy
- [Beyond Human Memory](docs/BEYOND_HUMAN_MEMORY.md) — cross-domain exploration from cellular autophagy to traditional knowledge frameworks

## License

[Apache 2.0](LICENSE)

## Links

- Website: [atagia.org](https://atagia.org)
- Author: Jordi Cor ([Acerting Art Inc.](https://acerting.com) / OjoCentauri)

---

Memory travels with the work. Stops at the door of what is not its own. Stays where it was lived.
