"""Small mechanical text helpers shared across engine modules."""

from __future__ import annotations


def strip_card_output_wrappers(text: str) -> str:
    """Remove complete presentation wrappers, preserving the value inside.

    Use only for machine-readable card fields, then validate their contract.
    Free text, names, and source quotations must not pass through this helper.
    """
    value = text.strip()
    pairs = (
        ('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019"),
        ("\u00ab", "\u00bb"), ("**", "**"), ("__", "__"), ("*", "*"), ("`", "`"),
    )
    while value:
        lines = value.splitlines()
        if (
            len(lines) >= 3
            and lines[0] in {"```", "```text", "```plaintext", "```json"}
            and lines[-1] == "```"
            and all("```" not in line for line in lines[1:-1])
        ):
            value = "\n".join(lines[1:-1]).strip()
            continue
        if len(lines) != 1:
            break
        for opening, closing in pairs:
            if (
                len(value) >= len(opening) + len(closing)
                and value.startswith(opening)
                and value.endswith(closing)
            ):
                inner = value[len(opening):-len(closing)]
                if opening not in inner and closing not in inner:
                    value = inner.strip()
                    break
        else:
            break
    return value


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
