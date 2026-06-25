"""Focused reliability + concurrency harness for the compactor SUMMARY card.

This benchmark drives the real engine method
``Compactor._summarize_message_ranges_card`` -- the per-range summary card
("card 2") -- in isolation, with FROZEN card-1 ranges, to measure reliability
(range coverage, retries, hard failures) and concurrency behavior (observed max
in-flight calls vs. the configured cap).

Prompt fidelity: the harness NEVER copies or re-implements the card 2 prompt. It
calls the engine method, so the prompt it measures is always the production
prompt. (See ``tests/benchmarks/test_benchmark_prompt_fidelity.py``.)
"""
