"""Visible source anchors and exact, snapshot-bound quote coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, model_validator


def source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SourceReference(BaseModel):
    """Code-produced coordinates, never character counts supplied by a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    char_start: int = Field(ge=0, strict=True)
    char_end: int = Field(gt=0, strict=True)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_ref: str | None = None
    end_ref: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> SourceReference:
        if self.char_end <= self.char_start:
            raise ValueError("Source reference must select a nonempty forward interval")
        return self

    def quote(self, source_text: str) -> str:
        if source_sha256(source_text) != self.source_sha256:
            raise ValueError("Source reference does not match the source snapshot")
        if self.char_end > len(source_text):
            raise ValueError("Source reference is outside the source text")
        return source_text[self.char_start : self.char_end]

    def rebase(
        self, *, chunk_text: str, source_text: str, chunk_start: int
    ) -> SourceReference:
        """Translate a verified chunk interval into its original message."""

        self.quote(chunk_text)
        if (
            chunk_start < 0
            or source_text[chunk_start : chunk_start + len(chunk_text)] != chunk_text
        ):
            raise ValueError("Extraction chunk is not an exact slice of the source")
        return SourceReference(
            char_start=chunk_start + self.char_start,
            char_end=chunk_start + self.char_end,
            source_sha256=source_sha256(source_text),
            start_ref=self.start_ref,
            end_ref=self.end_ref,
        )


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    reference_id: str
    char_start: int
    char_end: int


class SourceReferenceCatalog:
    """Label words and punctuation without interpreting the source language.

    Wide letters are individually addressable for scripts without word spaces.
    Combining marks stay attached to their preceding unit. Whitespace remains
    untouched between units and inside any selected range.
    """

    def __init__(self, source_text: str) -> None:
        self.source_text = source_text
        spans: list[tuple[int, int]] = []
        position = 0
        while position < len(source_text):
            char = source_text[position]
            if char.isspace():
                position += 1
                continue
            start = position
            position += 1
            word = char.isalnum() or char == "_"
            wide = unicodedata.east_asian_width(char) in {"W", "F"}
            while position < len(source_text):
                following = source_text[position]
                if unicodedata.category(following).startswith("M"):
                    position += 1
                elif (
                    word
                    and not wide
                    and (following.isalnum() or following == "_")
                    and unicodedata.east_asian_width(following) not in {"W", "F"}
                ):
                    position += 1
                else:
                    break
            spans.append((start, position))
        self.anchors = tuple(
            SourceAnchor(f"r{index}", start, end)
            for index, (start, end) in enumerate(spans, start=1)
        )
        self._by_id = {anchor.reference_id: anchor for anchor in self.anchors}
        self.source_hash = source_sha256(source_text)

    def render(self) -> str:
        """Render visible IDs; escape source delimiters so they cannot be IDs."""

        pieces: list[str] = []
        cursor = 0
        for anchor in self.anchors:
            pieces.append(html.escape(self.source_text[cursor : anchor.char_start]))
            literal = html.escape(self.source_text[anchor.char_start : anchor.char_end])
            literal = literal.replace("[", "&#91;").replace("]", "&#93;")
            pieces.append(f"[{anchor.reference_id}]{literal}")
            cursor = anchor.char_end
        pieces.append(html.escape(self.source_text[cursor:]))
        return "".join(pieces)

    def resolve(self, start_ref: str, end_ref: str) -> SourceReference:
        start = self._by_id.get(start_ref)
        end = self._by_id.get(end_ref)
        if start is None or end is None:
            raise ValueError("Unknown source reference; use only the displayed IDs")
        if start.char_start > end.char_start:
            raise ValueError("Source references must be in source order")
        return SourceReference(
            char_start=start.char_start,
            char_end=end.char_end,
            source_sha256=self.source_hash,
            start_ref=start_ref,
            end_ref=end_ref,
        )
