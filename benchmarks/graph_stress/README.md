# Graph Stress Set

A graph/relation stress benchmark for the Atagia memory engine. It targets the
retrieval patterns that similarity-only memory systems tend to miss:

- multi-hop relationship questions
- before/after event questions
- "who knows X" or "what connects X and Y" questions
- supersession/current-truth questions
- broad-list questions with dispersed evidence
- source-backed artifact/task questions

It is stored outside the default Atagia-bench dataset so routine Atagia-bench
baselines keep their historical question counts. The questions use the
Atagia-bench JSON contract, so the set runs with the existing Atagia-bench
runner by passing `--data-dir` and a holdout manifest.

## Dataset

As with the rest of Atagia-bench, the runner and adapter code are published but
the evaluation dataset (personas, conversations, questions, and the holdout
manifest) is private. This keeps the benchmark meaningful: models and retrieval
changes cannot overfit to questions they have never seen.

`tests/benchmarks/test_graph_stress_dataset.py` validates the dataset structure
(evidence turns present, evidence turn IDs resolving inside the synthetic
persona, all stress families represented) and requires the private dataset to
run.

## Running

With a dataset directory following the Atagia-bench JSON contract:

```bash
python -m benchmarks.atagia_bench \
  --data-dir <dataset-dir> \
  --holdout-file <dataset-dir>/holdout_v0.json \
  --benchmark-split development \
  --categories graph_stress \
  --provider anthropic \
  --model claude-sonnet-4-6
```
