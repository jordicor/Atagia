"""Centralized LLM output token limits and policy.

Atagia keeps every per-component cap on `max_output_tokens` in this file so
runtime, benchmark, and provider-diagnostic code share a single source of
truth.

Policy:

- An explicit `max_output_tokens` greater than 512 is honored as-is.
- An explicit `max_output_tokens` less than or equal to 512 is treated as
  "not specified" -- `LLMClient` passes None to the provider, which either
  uses an API-level default (OpenAI / OpenRouter / Gemini) or the explicit
  Anthropic fallback (`ANTHROPIC_FALLBACK_MAX_OUTPUT_TOKENS`).
- The 512 floor exists because tiny caps such as 256 silently break models
  with verbose structured output (e.g. Opus 4.6 used as a benchmark judge),
  so we prefer the provider default over a fragile micro-cap.

Standard cap is 8192. `max_tokens` is a ceiling, not a target, so a generous
cap costs nothing unless the model actually needs the headroom (e.g. for
reasoning tokens on Gemini 3.x or Claude thinking budgets). The handful of
larger values exist where structured-output extraction over long inputs
predictably needs more room.

The helper `apply_min_output_threshold` enforces the rule. It is applied
once at the entry point of `LLMClient.complete` and `LLMClient.complete_structured`
so every derived request inherits the cleaned value (schema fallback,
retries, provider routing).
"""

from __future__ import annotations


# === Provider-level fallbacks (used when no explicit value or value <=512) ===
ANTHROPIC_FALLBACK_MAX_OUTPUT_TOKENS = 8192

# === Runtime - memory components ===
MEMORY_EXTRACTION_MAX_OUTPUT_TOKENS = 16384
TEXT_CHUNKER_MAX_OUTPUT_TOKENS = 8192
COMPACTOR_CONVERSATION_CHUNK_MAX_OUTPUT_TOKENS = 8192
COMPACTOR_WORKSPACE_ROLLUP_MAX_OUTPUT_TOKENS = 8192
COMPACTOR_EPISODE_SYNTHESIS_MAX_OUTPUT_TOKENS = 8192
COMPACTOR_THEMATIC_PROFILE_MAX_OUTPUT_TOKENS = 8192
NEED_DETECTOR_MAX_OUTPUT_TOKENS = 8192
APPLICABILITY_SCORER_MAX_OUTPUT_TOKENS = 8192
CONTRACT_PROJECTION_MAX_OUTPUT_TOKENS = 8192
BELIEF_REVISER_MAX_OUTPUT_TOKENS = 8192
TOPIC_WORKING_SET_MAX_OUTPUT_TOKENS = 8192
CONTEXT_STALENESS_MAX_OUTPUT_TOKENS = 8192
CONSENT_CONFIRMATION_MAX_OUTPUT_TOKENS = 8192
INTENT_CLASSIFIER_STATEMENT_MAX_OUTPUT_TOKENS = 8192
INTENT_CLASSIFIER_CLAIM_KEY_MAX_OUTPUT_TOKENS = 8192
CONSEQUENCE_DETECTOR_MAX_OUTPUT_TOKENS = 8192
CONSEQUENCE_BUILDER_MAX_OUTPUT_TOKENS = 8192
METRICS_COMPUTER_MAX_OUTPUT_TOKENS = 8192
SUMMARY_PRIVACY_JUDGE_MAX_OUTPUT_TOKENS = 8192
SUMMARY_PRIVACY_REFINER_MAX_OUTPUT_TOKENS = 8192
EXPORT_ANONYMIZER_REWRITE_MAX_OUTPUT_TOKENS = 8192
EXPORT_ANONYMIZER_VERIFICATION_MAX_OUTPUT_TOKENS = 8192
CHAT_REPLY_MAX_OUTPUT_TOKENS = 8192

# === Benchmarks - answer generation ===
# Reasoning-capable models (e.g. Gemini 3.x with thinking enabled) consume the
# output budget on internal reasoning before emitting the answer, so caps must
# leave headroom for both reasoning + the final answer.
LOCOMO_ANSWER_MAX_OUTPUT_TOKENS = 8192
LOCOMO_REPLAY_PROBE_ANSWER_MAX_OUTPUT_TOKENS = 8192
ATAGIA_BENCH_ANSWER_MAX_OUTPUT_TOKENS = 8192
THIRD_PARTY_BENCH_ANSWER_MAX_OUTPUT_TOKENS = 8192

# === Benchmarks - judges and graders ===
GENERIC_JUDGE_MAX_OUTPUT_TOKENS = 8192
ATAGIA_BENCH_ABSTENTION_GRADER_MAX_OUTPUT_TOKENS = 8192
ATAGIA_BENCH_GATED_FACT_GRADER_MAX_OUTPUT_TOKENS = 8192
ATAGIA_BENCH_SUPERSESSION_GRADER_MAX_OUTPUT_TOKENS = 8192
COMPACTION_EVAL_JUDGE_MAX_OUTPUT_TOKENS = 8192
COMPACTION_EVAL_REFINER_MAX_OUTPUT_TOKENS = 8192

# === Benchmarks - provider diagnostic scripts ===
PROVIDER_BENCH_OLLAMA_WARMUP_MAX_OUTPUT_TOKENS = 10  # intentionally tiny; this path bypasses LLMClient and sends the literal to Ollama directly
PROVIDER_BENCH_OLLAMA_MAX_OUTPUT_TOKENS = 8192
PROVIDER_BENCH_OPENAI_MAX_OUTPUT_TOKENS = 8192
PROVIDER_BENCH_ANTHROPIC_MAX_OUTPUT_TOKENS = 8192
PROVIDER_BENCH_OPENROUTER_MAX_OUTPUT_TOKENS = 8192
PROVIDER_BENCH_GRANSABIO_MAX_OUTPUT_TOKENS = 8192
PROVIDER_BENCH_LLM_EVALUATOR_JUDGE_MAX_OUTPUT_TOKENS = 8192


def apply_min_output_threshold(requested: int | None) -> int | None:
    """Drop sub-threshold caps to None so provider default applies.

    Values <= 512 are treated as "unset" -- the LLMClient passes None to the
    provider, which either uses an API-level default (OpenAI/OpenRouter/Gemini)
    or our explicit Anthropic fallback (ANTHROPIC_FALLBACK_MAX_OUTPUT_TOKENS).

    This guard exists because tiny caps like 256 break models with verbose
    output (e.g., Opus 4.6 as a benchmark judge).
    """
    if requested is None:
        return None
    if requested <= 512:
        return None
    return requested
