"""Shared guard: a card prompt must not reuse its shadow benchmark's content.

The card prompts are graded by per-card shadow benchmarks (benchmarks/*_cards/
cases.jsonl). If a prompt's few-shot examples reuse a benchmark case's text or a
distinctive answer token, the benchmark can no longer measure generalization
(the prompt taught the model its own answer key). These helpers assert a built
prompt is disjoint from a benchmark's cases.

Two false-positive-safe signals are used:
  - sentence-like case strings (a space + length >= 15): an exact verbatim
    appearance in the prompt means an example mirrored a benchmark case.
  - distinctive tokens (code-like: digits/structure, or acronym/internal caps):
    unambiguous answer values like RIVER-19-BLUE, 8KLD219, aurora-main.

Limitation (documented on purpose): a pure Title-Case proper noun answer
(e.g. "Northstar") is indistinguishable from a sentence-initial common word
without a word list, so those are covered by an explicit ``regression_tokens``
list per call rather than the dynamic token scan.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PUNCT = ".,:;!?()[]{}\"'`"
# Structural ids (candidate_000, msg_assistant_1, tpc_42, cand_001) are shared
# vocabulary between prompts and runtime by design, not leaked content.
_STRUCTURAL_ID = re.compile(r"[a-z][a-z0-9]{0,15}_[a-z0-9_]+")


def _load_cases(rel_cases_path: str) -> list[dict]:
    text = (_PROJECT_ROOT / rel_cases_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _leak_strings_from_cases(cases: Iterable[object]) -> tuple[set[str], set[str]]:
    sentences: set[str] = set()
    tokens: set[str] = set()
    for case in cases:
        for raw in _walk_strings(case):
            stripped = raw.strip()
            if " " in stripped and len(stripped) >= 15:
                sentences.add(stripped)
            for word in stripped.split():
                if _is_distinctive_token(word):
                    tokens.add(word.strip(_PUNCT))
    return sentences, tokens


def _walk_strings(obj: object) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_strings(value)


def _is_distinctive_token(token: str) -> bool:
    text = token.strip(_PUNCT)
    if len(text) < 4:
        return False
    if _STRUCTURAL_ID.fullmatch(text):
        return False
    if "@" in text or "://" in text:
        return True
    # A distinctive answer value is code-like: it carries a digit alongside
    # letters or structure (8KLD219, RIVER-19-BLUE, +1-415-555-0199, $1,240).
    # Pure hyphenated lowercase words (one-off, third-party) and bare numbers
    # are NOT distinctive — they are common prose, not benchmark answer keys.
    if not any(ch.isdigit() for ch in text):
        return False
    has_alpha = any(ch.isalpha() for ch in text)
    has_structure = any(ch in text for ch in "-/$.,")
    return has_alpha or has_structure


def benchmark_leak_strings(rel_cases_path: str) -> tuple[set[str], set[str]]:
    """Return (sentence_like_strings, distinctive_tokens) for a benchmark's cases."""
    return _leak_strings_from_cases(_load_cases(rel_cases_path))


def _assert_no_leak(
    prompt_text: str,
    sentences: set[str],
    tokens: set[str],
    regression_tokens: Iterable[str],
) -> None:
    haystack = prompt_text.casefold()
    leaked_sentences = sorted(s for s in sentences if s.casefold() in haystack)
    assert not leaked_sentences, f"benchmark case text leaked into prompt: {leaked_sentences}"
    leaked_tokens = sorted(t for t in tokens if t.casefold() in haystack)
    assert not leaked_tokens, f"benchmark distinctive tokens leaked into prompt: {leaked_tokens}"
    leaked_regression = sorted(t for t in regression_tokens if t.casefold() in haystack)
    assert not leaked_regression, f"known-leaked tokens present in prompt: {leaked_regression}"


def assert_prompt_has_no_benchmark_leak(
    prompt_text: str,
    rel_cases_path: str,
    *,
    regression_tokens: Iterable[str] = (),
) -> None:
    """Assert the prompt reuses no case text / distinctive answer token."""
    sentences, tokens = benchmark_leak_strings(rel_cases_path)
    _assert_no_leak(prompt_text, sentences, tokens, regression_tokens)


def assert_prompt_has_no_benchmark_leak_in_cases(
    prompt_text: str,
    cases: Iterable[object],
    *,
    regression_tokens: Iterable[str] = (),
) -> None:
    """Same guard as ``assert_prompt_has_no_benchmark_leak`` for in-memory cases.

    Some shadow benchmarks (e.g. need_detection) define their cases as inline
    Python objects rather than a ``cases.jsonl`` file. The caller passes the
    JSON-serializable case dicts directly so the same disjointness scan applies.
    """
    sentences, tokens = _leak_strings_from_cases(cases)
    _assert_no_leak(prompt_text, sentences, tokens, regression_tokens)
