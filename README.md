# Atagia

*Memory that knows when to forget.*

Atagia is an open-source memory and perspective engine for AI systems that
interact with people across chats, devices, characters, projects, and worlds.
It is not a document-ingestion RAG engine. It is for medium- and long-term
assistant memory: useful continuity without treating every past token as
global, current, or equally relevant.

> "Atagia is memory for AIs, any kind of AI."

Atagia is pre-alpha and under active development. APIs and behavior may change.

An AI rarely lives in one place. The same voice can appear in a chat window, a
domestic robot, an NPC inside a game save, an agent running offline on a
laptop. Each one observes a different slice of the same person, the same
world, the same day. Each has to decide what to remember, what to forget, and
what belongs to a body, a world, or a voice it is not currently inhabiting.
Atagia is the layer underneath that decides, reading coordinates the host
has already declared, before any ranking happens.

Each candidate memory is scored on whether it actually applies to the
situation in front of it: the task, the active voice, the project, the body,
the world, the moment. Memories that no longer apply are not deleted. They
are recycled: kept as evidence of what once was, retired from current
answers. Atagia is named after autophagy, the cellular process of recycling
what no longer serves.

## Memory coordinates

Memory is not a flat pool. Before a memory is considered for ranking, Atagia
asks where it lives. Whose perspective owns it. Which voice was speaking when
it was captured. Which project, which body, which world it belongs to. These
are coordinates, declared by the host. They gate the candidate pool itself.

| Coordinate | The question it answers | What it makes possible |
|---|---|---|
| **User** | Whose memory is this? | Each user's memory is kept separate. |
| **Presence** | Which voice was speaking? | An accountant, a companion, and a character can share an AI without silently merging their identities. |
| **Space** | Which project, folder, room, or capsule? | A work project can have its own focus; a private folder can have its own boundary. |
| **Mind** | Whose perspective remembers this? | A user and an NPC can remember the same scene differently. Atagia keeps track of *who* remembers, not only *what*. |
| **Embodiment** | Which body or device captured this? | A drone's capabilities and observations stay attached to that body, rather than transferring to a home speaker. |
| **Realm** | Which world, simulation, or game save? | The kingdom's politics belong in the kingdom. The user's actual job need not enter the throne room. |

An *overseer* topology can read across many local Minds, Spaces, or Realms at
once, but only what has been explicitly granted. Local boundaries remain
intact. The overseer sees what it was given, labeled with where it came from.
Nothing inherits visibility by default.

**Mode** sets the style of retrieval: debugging can favor precise evidence,
an interview can favor broad recall, and a companion can draw on interaction
preferences.

The host can expose these as simple switches: remember across chats, remember
across devices, use an incognito chat, isolate this folder, bridge this Realm,
or grant an overseer a labeled view. These choices shape which memories
Atagia can bring into a conversation.

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

### Applicability, not just similarity

A memory can sound related and still be wrong for the moment. Atagia considers
whether it helps with the current request, belongs in the active context,
and is still current. Search finds candidates; applicability guides what
reaches the answer. Links to original messages help keep memories grounded
in what was actually said.

### Belief revision

New evidence can reinforce, revise, or retire a belief while preserving its
history. A belief like "user prefers detailed answers" does not get silently
replaced. It becomes "depth preference is
mode-dependent: concise for debugging, deep for research."

### Consequence chains

When a user reports the outcome of prior advice, Atagia records the chain:
action → outcome → tendency. These surface during retrieval when follow-up
failure or loop signals are detected.

### Keeping the conversation moving

Recent conversation, active topics, and prepared context help maintain
continuity. Atagia can reuse context when it still applies and retrieve fresh
memories when needed. Memory extraction runs in the background after messages
are stored.

### Storage

SQLite is the single source of truth. FTS5 handles lexical retrieval.
sqlite-vec is available as an optional semantic candidate-generation lane.
Redis accelerates queues and caching but is optional.

## Get started

Requires Python 3.12+ and access to a language model, hosted or local.

```bash
git clone https://github.com/jordicor/Atagia.git
cd Atagia
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

For a simple setup, set `ATAGIA_LLM_FORCED_GLOBAL_MODEL` in `.env` to your
chosen `provider/model` and fill in that provider's API key. Local model setup
and per-task model choices are covered in the
[configuration guide](docs/CONFIGURATION_REFERENCE.md#4-llm-model-selection).

### As a Python library

Save this as `example.py` and run `python example.py`. It stores a project
decision, waits for memory processing, then asks about it in another chat.

```python
import asyncio

from atagia import Atagia


async def main() -> None:
    async with Atagia(db_path="memory.db") as engine:
        await engine.create_user("user_1")
        await engine.create_conversation(
            "user_1", "planning",
            platform_id="demo",
            character_id="project_northstar",
            mode="coding_debug",
        )
        await engine.ingest_message(
            user_id="user_1",
            conversation_id="planning",
            role="user",
            text=(
                "For Project Northstar, remember that we are keeping SQLite "
                "for the first release."
            ),
        )
        if not await engine.flush(timeout_seconds=120):
            raise RuntimeError("Memory processing did not finish in time.")

        await engine.create_conversation(
            "user_1", "followup",
            platform_id="demo",
            character_id="project_northstar",
            mode="coding_debug",
        )
        result = await engine.chat(
            user_id="user_1",
            conversation_id="followup",
            message=(
                "Which database did we choose for Project Northstar's first release?"
            ),
        )
        print(result.response_text)


if __name__ == "__main__":
    asyncio.run(main())
```

### As a sidecar for your own LLM call

Already have a chat application? Use `engine.get_context()` to prepare memory
for your own model call, then `engine.add_response()` to store its reply.
Your application keeps control of generation.

For a ready-made integration, `SidecarBridge` handles context and response
persistence while allowing the host to continue if memory is unavailable.
See the [host integration guide](docs/HOST_SIDECAR_INTEGRATION.md).

### As an MCP server

```bash
pip install -e ".[mcp]"
```

Add Atagia to your MCP client's configuration. Replace the paths, model, and
provider-key placeholders with your settings:

```json
{
  "mcpServers": {
    "atagia-memory": {
      "command": "/path/to/.venv/bin/atagia-mcp",
      "env": {
        "ATAGIA_DB_PATH": "/path/to/memory.db",
        "ATAGIA_USER_ID": "desktop-user",
        "ATAGIA_PLATFORM_ID": "desktop",
        "ATAGIA_CONVERSATION_ID": "default-desktop-chat",
        "ATAGIA_<PROVIDER>_API_KEY": "your-api-key",
        "ATAGIA_LLM_FORCED_GLOBAL_MODEL": "provider/model"
      }
    }
  }
}
```

The tools let your client retrieve context, add and search memories, edit or
delete them, and manage conversations.

### As a REST API

After the setup above, keep `ATAGIA_SERVICE_MODE=true` in `.env`, set distinct
values for `ATAGIA_SERVICE_API_KEY` and `ATAGIA_ADMIN_API_KEY`, and enable
background memory processing with `ATAGIA_WORKERS_ENABLED=true`.

```bash
atagia-api --host 127.0.0.1 --port 8100
```

The API covers conversations, chat, memory retrieval, ingestion, editing, and
deletion. An OpenAI-compatible chat endpoint is also available for existing
clients. Keep service credentials on the server, not in browser code.
See the [API reference](docs/API.md) for routes and proxy setup.

### Configuration

Atagia is not tied to one model. You can use a single model for a simple setup,
choose different models for different tasks, or configure local inference.
Storage, model selection, and memory controls are covered in the
[configuration guide](docs/CONFIGURATION_REFERENCE.md).

## Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| API | FastAPI |
| Primary storage | SQLite + FTS5 |
| Inference | Configurable hosted and local models |
| Optional cache/queues | Redis |
| Optional semantic recall | sqlite-vec |

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Evaluation

Atagia is evaluated with LoCoMo and Atagia-bench, covering recall, belief
revision, cross-conversation memory, preferences, and multilingual use.
Evaluation is ongoing; current results are development signals, and public
baselines are not yet frozen.

## Research

- [Beyond Similarity: Applicability-Governed Memory](docs/Beyond_Similarity_Applicability_Governed_Memory.md): thesis paper with testable hypotheses and evaluation strategy
- [Beyond Human Memory](docs/BEYOND_HUMAN_MEMORY.md): cross-domain exploration from cellular autophagy to traditional knowledge frameworks

## License

[Apache 2.0](LICENSE)

## Links

- Website: [atagia.org](https://atagia.org)
- Author: Jordi Cor ([Acerting Art Inc.](https://acerting.com) / OjoCentauri)

---

Memory travels with the work. Stops at the door of what is not its own. Stays where it was lived.
