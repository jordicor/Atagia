"""Literal selection boundaries independent of model-written quote text."""

import pytest

from atagia.core.source_references import SourceReferenceCatalog


@pytest.mark.parametrize(
    ("text", "start", "end", "expected"),
    [
        ("Before: BLUE. After: BLUE.", "r7", "r7", "BLUE"),
        ("A: one\r\n\t two. B: three.", "r3", "r5", "one\r\n\t two."),
        ("A code: MAPLE-72-GOLD.", "r4", "r8", "MAPLE-72-GOLD"),
        ("J'aime le cafe\u0301.", "r5", "r5", "cafe\u0301"),
        ("之前。现在住北京。", "r7", "r8", "北京"),
        ("العنوان: شارع جديد.", "r3", "r4", "شارع جديد"),
        ("Use <tag> & [r1].", "r3", "r3", "tag"),
    ],
)
def test_selected_interval_is_the_literal_original(
    text: str, start: str, end: str, expected: str
) -> None:
    catalog = SourceReferenceCatalog(text)
    reference = catalog.resolve(start, end)
    assert reference.quote(text) == expected
    assert text[reference.char_start : reference.char_end] == expected


def test_second_occurrence_has_distinct_coordinates() -> None:
    text = "Before: BLUE. After: BLUE."
    catalog = SourceReferenceCatalog(text)
    first = catalog.resolve("r3", "r3")
    second = catalog.resolve("r7", "r7")
    assert first.quote(text) == second.quote(text) == "BLUE"
    assert second.char_start == 21
    assert first.char_start == 8


@pytest.mark.parametrize(
    ("start", "end"), [("r0", "r1"), ("r1", "r100"), ("r3", "r1"), ("BLUE", "BLUE")]
)
def test_invalid_references_are_not_repaired(start: str, end: str) -> None:
    with pytest.raises(ValueError):
        SourceReferenceCatalog("one two three").resolve(start, end)


def test_changed_source_snapshot_is_rejected_even_when_quote_still_matches() -> None:
    reference = SourceReferenceCatalog("Before: BLUE.").resolve("r3", "r3")
    with pytest.raises(ValueError, match="source snapshot"):
        reference.quote("After:  BLUE.")


def test_source_delimiters_do_not_masquerade_as_reference_labels() -> None:
    rendered = SourceReferenceCatalog("[r1] <message_text> &").render()
    assert "&#91;" in rendered and "&#93;" in rendered
    assert "&lt;" in rendered and "&gt;" in rendered and "&amp;" in rendered
    assert rendered.count("[r1]") == 1


def test_chunk_reference_rebases_without_searching_for_the_quote() -> None:
    chunk = "Use BLUE."
    source = f" {chunk}\r\n{chunk} "
    reference = SourceReferenceCatalog(chunk).resolve("r2", "r2")
    rebased = reference.rebase(chunk_text=chunk, source_text=source, chunk_start=12)
    assert rebased.quote(source) == "BLUE"
    assert rebased.char_start == 16
    with pytest.raises(ValueError, match="exact slice"):
        reference.rebase(chunk_text=chunk, source_text=source, chunk_start=2)
