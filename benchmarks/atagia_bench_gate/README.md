# Atagia-bench-gate

A small diagnostic suite for Atagia's **adaptive retrieval gate** — the per-turn
decision to skip memory retrieval when a turn does not depend on the user's
stored memory (general world knowledge, or something answerable from the recent
conversation window alone).

This suite is a behavioral probe, not an accuracy leaderboard. It exists to
answer two questions:

1. **Does the gate classify turns correctly?** Each turn is classified into one
   of four memory-dependence categories: `personal`, `conversation`, `world`,
   or `mixed`.
2. **Does the gate take the right action?** `world` and `conversation` turns
   should skip retrieval; `personal` and `mixed` turns must keep full
   retrieval. Skipping a turn that needed memory (a **false skip**) is the
   dangerous direction the gate must avoid.

## How it is built

The suite is made of synthetic, fictional personas. Each persona has a short
setup conversation that seeds a few concrete facts (a telescope model, a
supplier delivery day, a medication schedule, etc.), followed by probe
questions.

Most probes come in **pairs** that share a topic:

- a **world** variant answerable from general knowledge with no personal anchor
  (e.g., "Which planet has the most known moons?"), and
- a **personal** variant on the same topic that depends on the stored fact
  (e.g., "Which telescope model did I set up?").

Pairing makes a false skip directly comparable to a correct skip on the same
subject matter, so the suite separates "the gate is too eager" from "the gate
cannot tell the topic apart." A few **conversation-window** probes ask about
something just said in the visible recent window, and a couple of **mixed**
probes blend a personal anchor with a general request.

The suite includes non-English pairs (Spanish and French) so classification is
checked across languages, not just English phrasing.

Every question is labeled with its expected memory-dependence category and the
expected gate action. The expected action is derived from the category, not
hand-set per question, so labels stay consistent.

## What the runner measures

For each probe question the runner enables the adaptive retrieval gate, asks the
question through a real Atagia engine, and reads the gate's behavior from the
chat result's debug trace:

- the recorded **classification** (available even in shadow mode), and
- the **skip/retrieve action** the gate took (available once the engine reports
  a concrete gate status).

It then reports classification accuracy (overall and per language), gate-action
accuracy, and counts of false skips, correct skips, and missed skips. Anything
the engine does not report is left unscored rather than guessed.

## Running it

Validate the dataset and print its shape without any LLM calls:

```bash
python -m benchmarks.atagia_bench_gate --dry-run
```

Run the suite end to end with the gate enabled (requires LLM credentials in the
environment or via flags):

```bash
python -m benchmarks.atagia_bench_gate \
  --provider <provider> \
  --forced-global-model <model-id> \
  --output gate_report.json
```

Useful flags:

- `--dataset <path>`: use a custom gate-suite JSON dataset instead of the
  bundled `data/gate_suite_v0.json`.
- `--chat-model` / `--ingest-model` / `--retrieval-model`: route individual LLM
  components instead of forcing one global model.
- `--db-dir <dir>`: keep the per-persona databases (defaults to a temp dir).
- `--output <path>`: write the full JSON report.

The gate is enabled through the `ATAGIA_ADAPTIVE_RETRIEVAL` setting; the runner
sets it for the duration of the run and restores the previous value afterward.

## Files

- `dataset.py` — dataset models and loader (validates pairing and labels).
- `data/gate_suite_v0.json` — the synthetic paired-question dataset.
- `scoring.py` — pure scoring: reads the gate classification and action from a
  chat-result debug payload and aggregates accuracy.
- `runner.py` — end-to-end runner that ingests each persona and scores each
  probe question with the gate ON.
- `__main__.py` — CLI entry point.
