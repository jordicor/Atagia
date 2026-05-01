"""Helpers for safe embedding payload construction."""

from __future__ import annotations

from dataclasses import dataclass

from atagia.memory.intimacy_boundary_policy import is_restricted_intimacy_boundary


@dataclass(frozen=True, slots=True)
class EmbeddingUpsertPayload:
    """Resolved text and optional index text for an embedding upsert."""

    text: str
    index_text: str | None


def build_embedding_upsert_payload(
    *,
    canonical_text: str,
    index_text: str | None,
    privacy_level: int,
    intimacy_boundary: str = "ordinary",
    preserve_verbatim: bool,
) -> EmbeddingUpsertPayload:
    """Return a provider-safe embedding payload for one memory object."""
    if is_restricted_intimacy_boundary(intimacy_boundary):
        protected_index_text = str(index_text or "").strip()
        if not protected_index_text:
            raise ValueError(
                "Intimacy-bound memories require index_text for safe embedding generation"
            )
        return EmbeddingUpsertPayload(text=protected_index_text, index_text=None)
    if preserve_verbatim and privacy_level >= 2:
        protected_index_text = str(index_text or "").strip()
        if not protected_index_text:
            raise ValueError(
                "Protected verbatim memories require index_text for safe embedding generation"
            )
        return EmbeddingUpsertPayload(text=protected_index_text, index_text=None)
    return EmbeddingUpsertPayload(text=canonical_text, index_text=index_text)
