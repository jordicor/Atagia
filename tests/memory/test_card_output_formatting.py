"""Presentation cleanup preserves each card's value and validation rules."""

import pytest

from atagia.core.source_references import SourceReferenceCatalog
from atagia.memory.coverage_members_card import parse_members_output
from atagia.memory.evidence_cards import _parse_answer
from atagia.memory.extraction_cards import (
    parse_belief_key_output,
    parse_belief_value_output,
    parse_classification_output,
)
from atagia.memory.extraction_temporal import (
    parse_temporal_interval_output,
    parse_temporal_type_output,
)
from atagia.memory.language_profile import _parse_card_output
from atagia.memory.topic_working_set import (
    TopicUpdateActionType,
    _parse_content_field_output,
    _parse_route_card_output,
)


def test_extraction_scalar_fields_accept_wrappers() -> None:
    assert parse_classification_output("memory_kind", '**"evidence"**') == "evidence"
    assert parse_classification_output("memory_scope", "`user`") == "user"
    assert parse_classification_output("memory_confidence", '"0.85"') == 0.85
    assert parse_temporal_type_output('"bounded"') == "bounded"
    assert parse_belief_key_output("`workflow.edits.no_automatic_edits`") == "workflow.edits.no_automatic_edits"


@pytest.mark.parametrize("answer", ['"-0.8"', '"1.2"', "1. 0.99", "0.8 because it is clear"])
def test_confidence_does_not_drop_signs_or_extra_content(answer: str) -> None:
    with pytest.raises(ValueError):
        parse_classification_output("memory_confidence", answer)


@pytest.mark.parametrize("answer", ['"work flow.edits"', '"workflow.!edits"'])
def test_claim_keys_do_not_rewrite_inner_characters(answer: str) -> None:
    with pytest.raises(ValueError):
        parse_belief_key_output(answer)


def test_temporal_intervals_preserve_offsets_and_require_them() -> None:
    assert parse_temporal_interval_output(
        '```text\n"2026-03-29T10:00:00-04:00"\n`2026-03-29T11:00:00-04:00`\n```'
    ) == ("2026-03-29T10:00:00-04:00", "2026-03-29T11:00:00-04:00")
    with pytest.raises(ValueError, match="Invalid offset timestamp"):
        parse_temporal_interval_output('"2026-03-29T10:00:00"\nnone')


def test_reference_wrappers_never_change_the_quoted_source() -> None:
    source = 'O\'Neil said: "Do not change -0.8 or C++."'
    catalog = SourceReferenceCatalog(source)
    start, end = catalog.anchors[0].reference_id, catalog.anchors[-1].reference_id
    selected = _parse_answer("source_reference", f'"{start}" "{end}"', catalog)
    assert selected == catalog.resolve(start, end)
    assert selected.quote(source) == source
    assert _parse_answer("candidate_language", '```text\n"EN"\n\u201ces\u201d\n```', catalog) == ("en", "es")


def test_language_and_topic_decisions_accept_wrapped_tokens() -> None:
    languages, complete = _parse_card_output("observed", '```text\n"EN"\n\u201ces\u201d\n```')
    assert complete
    assert [item.language_code for item in languages.observed_user_languages] == ["en", "es"]
    routes = _parse_route_card_output(
        '\u201cupdate\u201d \u201ctopic_1\u201d \u201cmsg_1\u201d',
        valid_topic_ids={"topic_1"},
        valid_message_ids=("msg_1",),
    )
    assert len(routes) == 1
    assert routes[0].action is TopicUpdateActionType.UPDATE
    assert routes[0].target_id == "topic_1"
    assert routes[0].source_message_ids == ("msg_1",)


def test_free_text_and_json_member_names_retain_their_content() -> None:
    value = '"O\'Neil prefers C++"'
    assert parse_belief_value_output(value) == value
    assert _parse_content_field_output(
        value, field_name="title", action=TopicUpdateActionType.CREATE
    ) == value
    assert parse_members_output('"O\'Neil"\n"none"') == ["O'Neil", "none"]
