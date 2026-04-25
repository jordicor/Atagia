# Beyond Similarity: Applicability-Governed Memory for Personalized Dialogue Agents

**Status:** Research framing draft
**Date:** March 29, 2026
**Purpose:** Base document for a paper-oriented memory architecture, prior to the implementation roadmap
**Working shorthand:** **Atagia** (from “autophagy” — cellular self-recycling, one of the cross-domain research inspirations)
**Website:** atagia.org
**Companion document:** [Beyond Human Memory: A Cross-Domain Research Framework](BEYOND_HUMAN_MEMORY.md) — the exploratory research that inspired this architecture, covering mechanisms from biology, physics, neuroscience, and mythology. See Section 19 of this document for how the cross-domain inspirations map to computational mechanisms.

---

## Abstract

Long-horizon dialogue agents increasingly rely on external memory to preserve context across turns, sessions, and projects. Most current systems improve memory by optimizing what to store, how to compress it, or how to retrieve it through semantic similarity, hybrid search, graphs, or learned policies. These advances are real, but they leave a central problem insufficiently solved: **a memory can be relevant in content yet wrong for the current interaction**. In personalized assistants, the challenge is not only recall, but deciding **which memory is applicable now, for this user, in this mode, under this risk profile, with this conversational purpose**.

This document proposes **Applicability-Governed Memory** as a research framing and architecture hypothesis. The core idea is that memory selection should not be governed primarily by semantic similarity, but by **applicability**: the estimated likelihood that a memory will improve the current action under the active user state, assistant mode, task, scope, and epistemic conditions. This reframes memory from a storage-and-retrieval problem into a **selection-and-adaptation problem**.

The proposed framework introduces five main ideas:

1. **Applicability as a first-class control variable**, distinct from similarity.
2. **A multi-layer user memory model** separating evidence, revisable beliefs, compact user state, and interaction contract.
3. **Scoped personalization**, where cross-chat memory is filtered through explicit applicability boundaries rather than treated as a single global profile.
4. **Compaction as a view, not a source of truth**, so summaries help fit context windows without becoming the canonical knowledge base.
5. **Versioned, dependency-aware memory revision**, allowing the system to update, invalidate, and reinterpret prior beliefs instead of silently overwriting them.

The strongest publication path is not to claim a universal cognitive model, but to show that applicability-governed selection reduces irrelevant personalization, stale-memory failures, and compaction-induced drift while improving long-horizon collaboration.

---

## 1. Why this paper should exist

The memory problem in AI chat systems is often described too narrowly.

The usual framing is:

- context windows are finite,
- long conversations are expensive,
- summaries are lossy,
- retrieval misses important facts.

All of that is true. But for a personalized assistant, the deeper problem is this:

> **Even if the system remembers many true things about the user, it can still use the wrong memory at the wrong time.**

That is the failure mode behind many of the most visible product issues in real assistants:

- pulling an old preference after the user has changed,
- over-personalizing when the user wanted a neutral answer,
- injecting biographical context into a task that should stay local,
- carrying a behavior preference from one assistant mode into another where it no longer fits,
- using a summary artifact as if it were a grounded fact,
- retrieving something semantically nearby but pragmatically irrelevant.

Recent work has made this gap easier to articulate. Long-horizon memory systems are improving quickly, but recent benchmarks also show that current assistants still struggle with evolving user profiles, multi-session preference use, and irrational personalization. LoCoMo exposed the difficulty of very long conversational memory across many sessions; PersonaMem showed that models struggle to track dynamic user profiles; RPEval showed that personalization can hurt when irrelevant memories are injected; LifeBench pushed beyond explicit facts into procedural and habitual memory; and current surveys now describe memory as a write-manage-read loop tightly coupled to action rather than a passive store. These results suggest that memory quality is not only about retrieval strength, but about **decision quality over what should matter now**.

This document argues that the next meaningful step is to make **applicability** explicit.

---

## 2. Positioning against the current state of the art

The current landscape is strong enough that a new paper must be precise about what is and is not new.

### 2.1 What the field already does well

Recent systems have advanced the state of the art in several directions:

| Direction | Examples | What it improves |
|---|---|---|
| Multi-store memory | Mem0, AdaMem, MemoryOS, Letta | Separates recent context, user traits, events, and graphs |
| Graph-native / provenance-heavy memory | Graphiti, Kumiho | Tracks temporal facts, dependencies, provenance, and revision structure |
| Learned or structured memory control | AgeMem, A-MAC | Learns or optimizes store / retrieve / summarize / discard decisions |
| User-state adaptation | VARS and related personalization work | Biases retrieval with compact long-term and short-term user representations |
| Action-oriented memory | ActMem | Uses causal and semantic structures to support decisions, not only recall |
| Compaction / objectization | knowledge-object approaches, reflective memory managers | Reduces token cost and improves organization over raw logs |

These are important advances. Any new work that pretends the field is still at “just summarize old messages” would be outdated.

### 2.2 What still appears under-specified

Despite those advances, several gaps remain visible:

1. **Similarity still dominates too much of retrieval logic.**  
   Even when systems add graphs, rerankers, or user vectors, candidate generation and early scoring are still often heavily similarity-driven.

2. **Cross-session memory is often treated as one profile blob.**  
   In practice, the same user may interact with one system in different modes: coding, research, brainstorming, emotional support, roleplay, biography, decision support, etc. A fact or preference can be true yet only applicable within some of those modes.

3. **Many systems store preferences, but fewer model the interaction contract.**  
   What matters is not only *what the user likes*, but *how this user collaborates*: desired depth, tolerance for uncertainty, need for citations, pace, preferred correction style, appetite for pushback, and one-shot vs iterative workflow.

4. **Compaction is still often epistemically under-modeled.**  
   Summaries are operationally useful, but if a summary becomes the de facto truth layer, distortion accumulates and contradictions become hard to audit.

5. **Belief updating is often operational rather than epistemically explicit.**  
   A preference change, a corrected assumption, and a one-off exception should not be treated as the same event.

6. **Memory is still too often evaluated as recall instead of collaboration quality.**  
   The real product question is whether memory helps the system behave better over time for this user—not only whether it can answer a benchmark question.

This is the opening for a stronger paper.

---

## 3. Core thesis

### 3.1 Main claim

> **Memory selection for personalized dialogue agents should be governed primarily by applicability, not similarity.**

Similarity answers: “What looks like this query?”

Applicability answers: “What is worth bringing into this interaction now, given the user, the task, the mode, the scope, the current state, and the cost of being wrong?”

This distinction matters because many memory failures are not retrieval failures in the narrow sense. They are **policy failures**:

- the system retrieved something true but outdated,
- or true but scoped to another project,
- or true but private and unnecessary,
- or true but likely to derail the user’s intent,
- or true but too compressed to rely on,
- or true but based on a belief whose evidential base has changed.

### 3.2 The reframing

The usual question is:

> “How do we store and retrieve conversation history efficiently?”

The better question is:

> “How does the assistant decide which past information should influence the present action?”

That reframing changes the design target.

The key object is no longer “the memory database.”  
It is the **memory policy** that governs:

- admission,
- compaction,
- retrieval,
- revision,
- invalidation,
- and cross-scope transfer.

---

## 4. What “applicability” means

Applicability should be treated as a structured variable, not a vague synonym for relevance.

A memory is **applicable** when using it is expected to improve the current response or action more than omitting it would.

That expectation depends on multiple dimensions.

### 4.1 Applicability dimensions

A memory item can be scored along at least these axes:

- **Task fit** — does it matter for the task type at hand?
- **Mode fit** — does it apply in this assistant mode or persona?
- **Scope fit** — is it global, project-local, conversation-local, or ephemeral?
- **Temporal validity** — is it still likely to hold now?
- **Epistemic quality** — is it grounded in evidence, inferred, summarized, or highly transformed?
- **Dependency health** — do the assumptions this memory depends on still hold?
- **Risk relevance** — does it matter more when error cost is high?
- **Interaction fit** — does it affect how the assistant should collaborate, not only what it should say?
- **Privacy proportionality** — is invoking this memory proportionate to the current need?
- **Expected utility** — does it materially improve the next action?

### 4.2 A conceptual scoring function

A concrete implementation may vary, but the paper can define the idea formally:

```text
applicability(memory, interaction)
  = f(task_fit,
      mode_fit,
      scope_fit,
      temporal_validity,
      epistemic_quality,
      dependency_health,
      risk_relevance,
      interaction_fit,
      privacy_proportionality,
      expected_utility)
```

Similarity may still contribute to candidate generation, but it should not be the governing principle.

### 4.3 Why applicability is not just “better reranking”

This framing is stronger than a standard reranker for three reasons:

1. **It is scope-aware.**  
   The same user fact may be applicable in one mode and inapplicable in another.

2. **It is epistemically aware.**  
   A memory summary, an inferred preference, and a verbatim user statement are not treated as equivalent evidence.

3. **It is interaction-aware.**  
   The system is not only retrieving content for an answer, but deciding how to collaborate.

---

## 5. The proposed framework: Atagia

Atagia is a conceptual architecture for long-horizon personalized dialogue systems.

It is not one single storage trick. It is a **governing structure** for how memory objects are represented, selected, revised, and composed.

### 5.1 Four memory layers

#### A. Evidence layer

This is the closest layer to what actually happened.

It contains:

- verbatim spans,
- extracted events,
- structured conversation facts,
- citations to source messages,
- timestamps,
- provenance,
- and confidence about extraction.

The evidence layer should be append-heavy and hard to mutate silently.

Its role is to anchor the rest of the system.

#### B. Belief layer

This stores **interpretations** of evidence.

Examples:

- “The user prefers concise answers for debugging tasks.”
- “The user is likely in research mode when asking open-ended comparison questions.”
- “The user dislikes repeated disclaimers.”
- “Project X uses a fragile legacy stack; aggressive refactors often backfire.”

Beliefs are not raw facts. They are **revisable claims** derived from evidence.

They should therefore be:

- versioned,
- linked to supporting evidence,
- linked to contradicting evidence,
- scored for stability,
- and capable of being superseded rather than overwritten.

#### C. User-state layer

This is a compact, continuously updated representation of the user’s stable and transient state.

It may include:

- stable cross-session preference signals,
- short-term temporary focus,
- current topical activation,
- recent frustration / uncertainty / urgency signals,
- coarse collaboration mode probabilities.

This layer is not the source of truth. It is a **biasing layer** for retrieval and response planning.

#### D. Interaction contract layer

This is one of the most important additions.

The interaction contract is the assistant’s current best model of **how this user wants to work with the system**.

It can include dimensions such as:

- desired depth,
- tolerance for ambiguity,
- appetite for citations,
- preference for iterative collaboration vs one-shot delivery,
- preference for pushback vs alignment,
- preferred tone and directness,
- tolerance for proactive suggestions,
- response pacing,
- preference for conceptual framing vs implementation-first answers.

This layer does not only change content selection. It changes the **style of cognition the assistant should enact**.

That makes it distinct from standard preference memory.

---

## 6. Scoped personalization: one user, many applicability zones

A core requirement for real deployment is that cross-chat memory should not collapse into one monolithic profile.

The same person may use the assistant in very different ways.

A practical system needs a scope model.

### 6.1 Recommended scopes

Each memory object should carry an explicit scope:

- **Global user** — durable traits or preferences that apply across most interactions.
- **Assistant mode** — applicable to a class of prompts or personas (e.g. coding assistant, research assistant, biography interviewer, companion mode).
- **Workspace / project** — tied to one project, repository, client, or objective.
- **Conversation** — relevant only within a particular thread.
- **Ephemeral session** — temporary conditions that should fade quickly unless reinforced.

### 6.2 Why scope is not just metadata

Scope must actively constrain retrieval.

For example:

- A global preference for rigorous reasoning may transfer widely.
- A preference for emotional gentleness may matter in companion mode but not in a formal evaluation workflow.
- A project-specific preference for Python typing conventions should not leak into unrelated chats.
- A temporary preference expressed under stress should not immediately overwrite a stable long-term trait.

This is where applicability becomes operational.

A memory can be true, recent, and semantically similar—and still be rejected because its **scope fit is poor**.

---

## 7. Need-triggered retrieval instead of query-only retrieval

A second central component is the move from query-triggered retrieval to **need-triggered retrieval**.

Most memory systems wake up when the query resembles stored content.

That is useful but incomplete.

Some of the most valuable retrieval moments happen when the system detects not textual similarity, but **interactional need**.

### 7.1 Retrieval triggers that are not just semantic

Need signals can include:

- contradiction between current user request and stored stable beliefs,
- user correction patterns,
- ambiguity about referents or plans,
- repeated loops or stalled progress,
- sudden rise in risk or error cost,
- frustration markers,
- follow-up questions that imply prior failure,
- requests that are under-specified but high impact,
- mode shifts.

### 7.2 Why this matters

In many practical chats, the crucial memory is not the one that best matches the current phrasing.

It is the one that helps prevent a likely mistake.

This is especially important in:

- debugging,
- planning,
- emotionally sensitive interactions,
- long-running projects,
- and any domain where the assistant must adapt to the user’s habits and failure patterns.

This idea can be framed as a formal policy contribution:

> **Retrieve when needed, not only when matched.**

---

## 8. Evidence, belief, and revision

A robust memory system must distinguish between **what was observed** and **what is currently believed**.

This separation is essential for both scientific clarity and product reliability.

### 8.1 The problem with silent overwrite

Suppose a user previously said they like highly detailed responses. Months later they repeatedly ask for speed and brevity.

A simplistic memory system might just replace one preference with the other.

That loses crucial information:

- Was the earlier belief wrong?
- Did the user genuinely change?
- Is the new preference only temporary?
- Does the preference depend on task type?
- Does the earlier preference still apply in learning mode?

### 8.2 A better model

The system should preserve:

- the original evidence,
- the old belief revision,
- the new belief revision,
- the relation between them,
- and the conditions under which each seems to apply.

This yields a more faithful representation:

- **Evidence**: “In three research conversations the user explicitly requested more depth.”
- **New evidence**: “In five debugging conversations this month the user repeatedly preferred short actionable answers.”
- **Current belief**: “Depth preference is mode-dependent: deep for research, concise for debugging.”

That is not just memory. It is adaptive interpretation.

### 8.3 Why belief revision matters for publication value

This creates a strong bridge between practical assistant memory and formal work on belief revision and truth maintenance.

That bridge is more publishable than broad metaphor alone because it yields:

- auditable updates,
- explicit supersession,
- contradiction handling,
- dependency tracing,
- and measurable error reduction.

---

## 9. Dependency-aware invalidation

Another underused but powerful idea is to treat memories as dependent structures rather than isolated entries.

A belief may depend on:

- specific evidence,
- another belief,
- a mode assignment,
- a project context,
- a time interval,
- or an inferred user state.

When one of those dependencies changes, the system should not continue using downstream beliefs with full confidence.

### 9.1 Example

If the system has learned:

- “User prefers architecture discussions before code,”

but that belief was strongly supported by past conversations in exploratory planning mode, then it may become less applicable when the user enters urgent debugging mode.

The preference is not false globally. Its applicability basis has changed.

### 9.2 Proposed behavior

When a dependency is weakened or invalidated, dependent memories should be marked as:

- unstable,
- scope-limited,
- requiring review,
- or temporarily down-ranked.

This is important for avoiding stale personalization and for making the system’s adaptation path auditable.

---

## 10. Compaction as a view, not as truth

This is one of the most practically important design principles in the document.

### 10.1 The compaction trap

Production systems need compaction.

Without it:

- token costs grow,
- latency grows,
- and context quality degrades.

But if summaries become the canonical truth layer, then the system accumulates distortion:

- details disappear,
- ambiguous claims harden into facts,
- causal links vanish,
- exceptions are erased,
- and summary-of-summary drift compounds over time.

Recent work has made this problem explicit: compaction can be operationally necessary while still being catastrophically lossy if treated as the memory source of truth.

### 10.2 Proposed principle

> **Compaction should optimize transport into context, not redefine what the system knows.**

That implies:

- summaries are views,
- evidence remains canonical,
- beliefs remain revisable and traceable,
- and high-stakes actions should be able to re-ground from evidence when needed.

### 10.3 Consequence for architecture

A future implementation should support at least three distinct objects:

- **canonical evidence objects**,
- **belief objects**,
- **context views** generated for current inference.

This separation is crucial for both product reliability and paper clarity.

---

## 11. Interaction contract memory

This is likely one of the most genuinely differentiated ideas in the whole framework.

Many memory systems store facts about the user. Fewer store the **collaboration protocol** that emerges over time.

That collaboration protocol is often the difference between a mediocre assistant and one that feels deeply adapted.

### 11.1 What belongs here

Examples include:

- “Challenge me when my assumptions look weak.”
- “Do not over-explain obvious steps when I am coding.”
- “When I ask for research, give citations and differentiate fact from inference.”
- “In emotional conversations, do not become coldly analytical too fast.”
- “When I brainstorm, do not force premature closure.”
- “I often think out loud before I know what I want; help me shape it.”

These are not mere preferences. They are **interaction rules**.

### 11.2 Why this matters

The same factual memory may be useless if the assistant applies it with the wrong collaboration style.

Interaction contract memory directly addresses the user experience question:

> “Does the assistant understand not only me, but how to work with me?”

This also aligns with real product usage, where the assistant may serve as:

- search interface,
- collaborator,
- coach,
- companion,
- interviewer,
- debugger,
- or research partner.

The contract should vary by mode, and its applicability should be governed accordingly.

---

## 12. Consequence chains and outcome memory

Another promising layer is to remember not only facts and preferences, but **what led to what**.

This is especially valuable for systems meant to improve over time with a specific user.

### 12.1 The core idea

Store patterns such as:

- recommendation → user outcome,
- clarification strategy → successful resolution,
- response style → user correction,
- action plan → later failure,
- exploratory questioning → deeper disclosure,
- overly broad answer → user frustration.

### 12.2 Why consequence memory matters

This shifts the memory system from “user profile storage” toward **adaptive collaboration memory**.

The assistant does not only remember the user.  
It remembers **what tends to work with the user**.

That creates a clearer path to measurable improvement because it affects future action policy directly.

---

## 13. A minimal memory object model

To make the paper concrete without yet becoming a full technical design, the framework can define a minimal memory object schema.

```yaml
memory_object:
  id: unique identifier
  layer: evidence | belief | user_state | interaction_contract | consequence
  scope: global_user | assistant_mode | workspace | conversation | ephemeral_session
  content: human-readable or structured payload
  evidence_refs: supporting evidence ids
  confidence: extraction or inference confidence
  stability: how likely the memory is to remain valid over time
  vitality: use/reinforcement signal
  applicability_tags: task, mode, domain, risk, privacy, temporal tags
  dependencies: upstream evidence, beliefs, scopes, or states
  supersedes: prior memory ids if this revises earlier beliefs
  valid_from: timestamp or interval
  valid_to: optional timestamp or interval
  epistemic_status: observed | inferred | summarized | speculative
  distortion_score: distance from original evidence
  review_after: timestamp for re-check or decay
```

This object model already implies several design principles:

- not all memories are of the same type,
- not all memories are equally trustworthy,
- not all memories apply everywhere,
- and not all memories should survive unchanged.

---

## 14. The inference loop

A publishable architecture becomes easier to understand when the inference loop is explicit.

```text
Incoming message
  -> mode resolver
  -> need estimator
  -> candidate retrieval across scopes and layers
  -> applicability scoring and gating
  -> evidence / belief / contract composition
  -> response planning
  -> response generation
  -> post-hoc write / revise / strengthen / weaken
```

### 14.1 Step descriptions

- **Mode resolver** estimates the current assistant mode or collaboration frame.
- **Need estimator** decides whether memory retrieval is necessary and how deep it should go.
- **Candidate retrieval** gathers memory candidates from relevant scopes and layers.
- **Applicability scoring** filters and reranks candidates by present usefulness, not mere similarity.
- **Composition** constructs the context package for the model.
- **Post-hoc memory update** decides what new evidence, beliefs, contract shifts, or consequences should be stored or revised.

This makes memory an active policy loop rather than passive recall.

---

## 15. The strongest publishable contributions

To maximize paper strength, the work should make a narrow set of claims very well.

### 15.1 Contribution A — Applicability-governed retrieval

A formal and practical framework where semantic similarity is only one signal among others, and final memory use is governed by applicability under task, mode, scope, and epistemic constraints.

### 15.2 Contribution B — Interaction contract memory

A dedicated memory layer for how the user prefers to collaborate, distinct from factual preference memory.

### 15.3 Contribution C — Evidence / belief separation for personalized agents

A memory representation that distinguishes observed evidence from revisable user-model beliefs, enabling explicit update paths instead of silent overwrite.

### 15.4 Contribution D — Compaction without epistemic collapse

A design principle and implementation strategy where summaries are context views rather than canonical truth.

### 15.5 Contribution E — Scoped cross-chat personalization

A scope-aware system where cross-session memory is not globally applied by default, but filtered through applicability boundaries.

These five together form a much stronger research story than “bio-inspired memory” as a standalone claim.

---

## 16. What this paper should *not* claim

To stay rigorous, the paper should avoid several over-claims.

### 16.1 Avoid claiming that no one has done layered memory

That is no longer true.

### 16.2 Avoid claiming that metaphor alone is novelty

Biology, physics, and symbolic systems can be powerful design inspiration, but they are not sufficient as the main scientific contribution.

### 16.3 Avoid claiming perfect personal understanding

The system should be presented as maintaining revisable, scoped, probabilistic models of the user—not as “knowing” the person in a total sense.

### 16.4 Avoid treating every remembered item as equally desirable

Sometimes the best memory behavior is non-use, down-weighting, or deliberate non-transfer.

### 16.5 Avoid making the first paper too broad

A paper that tries to introduce a total cognitive architecture, a new benchmark suite, a full product stack, and a philosophical theory of memory at once will likely lose focus.

---

## 17. Research hypotheses

A good paper should commit to testable hypotheses.

### H1 — Applicability-governed retrieval reduces irrelevant personalization

Compared with similarity-dominant memory selection, applicability-governed selection will reduce cases where user memory is injected but harms intent understanding.

### H2 — Interaction contract memory improves collaboration quality

Explicit memory of collaboration style will reduce user correction rate and increase perceived alignment in long-horizon interactions.

### H3 — Evidence / belief separation reduces stale-memory errors

Versioned belief revision linked to evidence will outperform overwrite-style memory on evolving preference benchmarks.

### H4 — Scope-aware memory transfer outperforms global-profile transfer

Applying memory through explicit scopes will improve both task success and user satisfaction by reducing inappropriate cross-chat leakage.

### H5 — Non-canonical compaction preserves utility under token pressure

Treating summaries as generated context views rather than truth objects will reduce compaction-induced drift and contradiction failures in long sessions.

---

## 18. Evaluation strategy

A convincing paper will need evaluation on both existing benchmarks and targeted new probes.

### 18.1 Existing benchmarks to use

- **LoCoMo** — long-horizon conversational memory and reasoning.
- **PersonaMem** — evolving user profile tracking.
- **RPEval** — irrational personalization and selective memory use.
- **MultiSessionCollab** — long-horizon preference learning through repeated collaboration.
- **LifeBench** — long-horizon multi-source memory, including non-declarative signals.
- **EverMemBench / related recent benchmarks** — memory awareness and profile understanding.

### 18.2 Metrics that matter

The paper should go beyond accuracy alone.

Recommended metrics:

- task success / answer quality,
- irrelevant personalization rate,
- user correction rate,
- stale-belief error rate,
- contradiction rate after profile updates,
- token cost per successful answer,
- compaction drift score,
- dependency invalidation precision,
- retrieval precision by scope,
- recovery quality after preference change.

### 18.3 New diagnostic evaluations worth introducing

A strong paper may need one or two new diagnostic slices, even if not a full benchmark release.

#### A. Scope collision tests

Check whether global, mode-local, and project-local memories are used only when appropriate.

#### B. Contract mismatch tests

Measure whether the assistant uses correct collaboration style even when factual retrieval is correct.

#### C. Belief revision conflict tests

Inject controlled preference changes and exceptions, then measure whether the system updates gracefully rather than oscillating or overwriting blindly.

#### D. Compaction trust tests

Compare answer quality and factual stability when the system relies on raw evidence, summaries as views, or summary-as-truth baselines.

---

## 19. Why the original “Beyond Human Memory” research still matters

The original broader research direction remains valuable, but its role should be reframed.

Its strongest use is not as the main novelty claim.  
Its strongest use is as a **design inspiration appendix** that explains where specific mechanisms came from.

### 19.1 Design inspirations that remain useful

| Inspiration | Computational translation |
|---|---|
| Danger theory | Need-triggered retrieval based on conversational risk or distress |
| Reconsolidation | Retrieved beliefs enter an update window and may be revised |
| Autophagy | Low-value memories can be decomposed and recycled into surviving structures |
| Diaschisis | Dependency-aware invalidation of connected memories |
| Phase transitions / quorum | Beliefs crystallize only after sufficient repeated evidence |
| Maya / illusion | Track distortion between evidence and transformed memory views |
| Songlines / non-linear memory | Organize retrieval partly by narrative path, not only time |
| Somatic markers | Use lightweight affective / friction signals as early policy bias |

These ideas can be very powerful in the narrative around the research, especially for talks, essays, brand positioning, and future exploratory papers.

But for the first serious paper, they should support the architecture—not replace it.

---

## 20. Recommended paper story

The best paper story is likely this:

1. Long-context alone does not solve personalization.
2. Memory systems are improving, but still over-rely on similarity and under-model applicability.
3. Personalized dialogue requires separating evidence, revisable beliefs, compact user state, and interaction contract.
4. Cross-chat memory must be scope-aware.
5. Compaction must not collapse epistemic structure.
6. Applicability-governed retrieval improves long-horizon collaboration quality.

That story is:

- concrete,
- timely,
- grounded in recent literature,
- implementable,
- benchmarkable,
- and differentiated enough to be interesting.

---

## 21. How this should map to implementation later

This document is intentionally pre-technical, but it should still imply an implementable direction.

A later technical document can map the framework onto a pragmatic stack such as:

- **Python + FastAPI** for orchestration and APIs,
- **Redis** for hot working state, queues, and active memory windows,
- **SQLite** for canonical evidence and belief storage, full-text search, and lightweight local deployment,
- **optional Neo4j** for dependency-heavy graph operations if the value justifies the complexity.

The key design principle is that storage technology follows the memory ontology—not the other way around.

---

## 22. Practical examples

### Example 1 — Same user, different modes

The user asks in one conversation:

> “For research, please be exhaustive and cite sources.”

Later, in a debugging conversation, the user says:

> “Just tell me the fastest fix.”

A naive memory system may average these into inconsistency.

An applicability-governed system should learn:

- exhaustive + cited in research mode,
- concise + action-first in debugging mode.

Both are true. Neither should overwrite the other.

### Example 2 — Preference update vs exception

The user historically prefers challenge and pushback. During a stressful conversation they request a gentler tone.

The system should not instantly erase the stable preference. It should represent:

- stable contract trait: appreciates pushback,
- temporary state: currently needs lower-friction interaction,
- applicability rule: gentler tone under elevated stress signals.

### Example 3 — Cross-project leakage

The user has one codebase where aggressive refactors are welcome, and another legacy system where they are risky.

A global memory item such as “user likes bold refactors” is not specific enough.

The system should preserve project-local consequence memory:

- in project A: bold refactors often help,
- in project B: bold refactors repeatedly caused regressions.

The difference is not recall quality. It is scope discipline.

---

## 23. Strong title candidates

Primary candidate:

- **Beyond Similarity: Applicability-Governed Memory for Personalized Dialogue Agents**

Alternative candidates:

- **Beyond Retrieval: Scoped and Revisable Memory for Personalized Dialogue Agents**
- **Applicability over Similarity: A Memory Architecture for Long-Horizon Personalized Assistants**
- **Versioned User Memory: Applicability-Governed Personalization for Long-Horizon Assistants**
- **Compaction Without Forgetting: Applicability-Governed Memory for Long-Horizon Dialogue**

---

## 24. Final synthesis

The strongest version of this research is not a claim that AI should imitate every form of memory found in biology, cosmology, or myth.

It is a narrower and more powerful claim:

> **Personalized AI memory fails when it treats “similar” as a substitute for “applicable.”**

What matters is not only whether the assistant can retrieve the past.

What matters is whether it can decide:

- which past matters now,
- at what level of trust,
- within which scope,
- under which interaction contract,
- and with what revision history.

That is the real bridge between long-term memory and adaptive collaboration.

If implemented well, Applicability-Governed Memory could contribute in three directions at once:

- a better architecture for real assistants,
- a stronger research contribution than generic “memory systems,”
- and a clearer public narrative about what makes persistent AI genuinely useful.

---

## References and anchor sources

### Surveys and framing

- Du, P. (2026). *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers*. arXiv:2603.07670. https://arxiv.org/abs/2603.07670

### Benchmarks and evaluation

- Maharana, A. et al. (2024). *Evaluating Very Long-Term Conversational Memory of LLM Agents*. LoCoMo. arXiv:2402.17753. https://arxiv.org/abs/2402.17753
- Jiang, B. et al. (2025). *Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale*. PersonaMem. arXiv:2504.14225. https://arxiv.org/abs/2504.14225
- Feng, X. et al. (2026). *How Does Personalized Memory Shape LLM Behavior? Benchmarking Rational Preference Utilization in Personalized Assistants*. RPEval. arXiv:2601.16621. https://arxiv.org/abs/2601.16621
- Mehri, S. et al. (2026). *Learning User Preferences Through Interaction for Long-Term Collaboration*. MultiSessionCollab. arXiv:2601.02702. https://arxiv.org/abs/2601.02702
- Cheng, Z. et al. (2026). *LifeBench: A Benchmark for Long-Horizon Multi-Source Memory*. arXiv:2603.03781. https://arxiv.org/abs/2603.03781

### Memory architectures and policies

- Yan, S. et al. (2026). *AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents*. arXiv:2603.16496. https://arxiv.org/abs/2603.16496
- Yu, Y. et al. (2026). *Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents*. AgeMem. arXiv:2601.01885. https://arxiv.org/abs/2601.01885
- Zhang, G. et al. (2026). *Adaptive Memory Admission Control for LLM Agents*. A-MAC. arXiv:2603.04549. https://arxiv.org/abs/2603.04549
- Park, Y. B. (2026). *Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision Semantics for Versioned Memory Architectures*. Kumiho. arXiv:2603.17244. https://arxiv.org/abs/2603.17244
- Zhang, X. et al. (2026). *ActMem: Bridging the Gap Between Memory Retrieval and Reasoning in LLM-Based Agents*. arXiv:2603.00026. https://arxiv.org/abs/2603.00026
- Hao, Y. et al. (2026). *User Preference Modeling for Conversational LLM Agents: Weak Rewards from Retrieval-Augmented Interaction*. VARS. arXiv:2603.20939. https://arxiv.org/abs/2603.20939
- *Knowledge Objects for Persistent LLM Memory* (2026). arXiv:2603.17781. https://arxiv.org/abs/2603.17781

### Systems and infrastructure

- Mem0 documentation: https://docs.mem0.ai/introduction
- Mem0 GitHub: https://github.com/mem0ai/mem0
- Graphiti GitHub: https://github.com/getzep/graphiti
- Letta GitHub: https://github.com/letta-ai/letta
- SQLite FTS5 documentation: https://www.sqlite.org/fts5.html
- sqlite-vec GitHub: https://github.com/asg017/sqlite-vec

### Belief revision and truth maintenance

- Doyle, J. (1979). *A Truth Maintenance System*. Artificial Intelligence, 12(3), 231–272. https://www.sciencedirect.com/science/article/pii/0004370279900080
- Hansson, S. O. *Logic of Belief Revision*. Stanford Encyclopedia of Philosophy. https://plato.stanford.edu/entries/logic-belief-revision/

---

## Suggested next step

After reviewing this framing, the next document should translate it into a technical design with:

- memory schemas,
- retrieval and revision flows,
- Redis / SQLite / optional Neo4j mapping,
- API boundaries,
- background jobs,
- and phased MVP implementation.
