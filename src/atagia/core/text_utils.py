"""Small mechanical text helpers shared across engine modules."""

from __future__ import annotations


def truncate_inline(text: str, max_chars: int) -> str:
    """Collapse whitespace and bound length for inline rendering.

    Used wherever LLM-derived text is surfaced into a prompt verbatim (coverage
    display labels, composer inline values): it must be single-line and bounded
    the same way at every site, so this is the single source of truth.
    """
    if max_chars <= 0:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return f"{normalized[: max_chars - 3].rstrip()}..."
