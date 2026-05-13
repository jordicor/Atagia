# Graph Stress Set

This folder contains the Wave 0 graph/relation stress set from
`docs/PROPOSAL_retrieval_breakthrough_experiments.md`.

It is intentionally stored outside the default Atagia-bench dataset so routine
Atagia-bench baselines keep their historical question counts. The data uses the
Atagia-bench JSON contract, so it can be run with the existing Atagia-bench
runner by passing `--data-dir` and the local holdout manifest.

## Validate Structure

```bash
pytest tests/benchmarks/test_graph_stress_dataset.py
```

The validation checks that every graph-stress question has evidence turns, that
the evidence turn IDs resolve inside the synthetic persona, and that all Wave 0
stress families are represented:

- multi-hop relationship questions
- before/after event questions
- "who knows X" or "what connects X and Y" questions
- supersession/current-truth questions
- broad-list questions with dispersed evidence
- source-backed artifact/task questions

## Run The Stress Set

Development slice:

```bash
python -m benchmarks.atagia_bench \
  --data-dir benchmarks/graph_stress/data \
  --holdout-file benchmarks/graph_stress/data/holdout_v0.json \
  --benchmark-split development \
  --categories graph_stress \
  --provider anthropic \
  --model claude-sonnet-4-6
```

Holdout slice:

```bash
python -m benchmarks.atagia_bench \
  --data-dir benchmarks/graph_stress/data \
  --holdout-file benchmarks/graph_stress/data/holdout_v0.json \
  --benchmark-split holdout \
  --categories graph_stress \
  --provider anthropic \
  --model claude-sonnet-4-6
```

Attach the Atagia-bench report, run manifest, failed-question custody report,
failure taxonomy report, and diff against the frozen baseline before accepting a
retrieval experiment that claims to improve this set.

## Frozen Baseline

The first baseline run was frozen on 2026-05-08 with current default retrieval
behavior, embeddings disabled, Anthropic Sonnet answer generation, and Opus
judging:

- Report: `benchmarks/results/graph_stress/atagia-bench-report-20260508T163952Z.json`
- Run manifest: `benchmarks/results/graph_stress/atagia-bench-run-manifest-20260508T163952Z.json`
- Failed-question custody: `benchmarks/results/graph_stress/atagia-bench-failed-custody-20260508T163952Z.json`
- Failure taxonomy: `benchmarks/results/graph_stress/atagia-bench-failure-taxonomy-20260508T163952Z.json`

Result: `5/8 = 62.5%`, average score `0.600`, `0` critical errors. The three
failures were all `retrieval_or_ranking_miss` / `retrieval_insufficient`
questions. There is no diff artifact for the initial baseline; future runs
should pass this report as `--diff-against`.

## Validated Baseline

Wave 0.5 found that the initial failures depended on chat-scoped or future-valid
seed data. The graph-stress set was corrected without relaxing retrieval policy,
and the validated baseline is now:

- Report: `benchmarks/results/graph_stress_wave05/atagia-bench-report-20260508T204542Z.json`
- Run manifest: `benchmarks/results/graph_stress_wave05/atagia-bench-run-manifest-20260508T204542Z.json`
- Failed-question custody: `benchmarks/results/graph_stress_wave05/atagia-bench-failed-custody-20260508T204542Z.json`
- Failure taxonomy: `benchmarks/results/graph_stress_wave05/atagia-bench-failure-taxonomy-20260508T204542Z.json`

Result: `8/8 = 100.0%`, average score `0.975`, `0` critical errors, and no
failed questions. Use this validated baseline for future graph-stress
no-regression gates.
