"""Lossless proxy tool transcript, projection, and fingerprint tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from atagia.services.proxy_transcript import (
    ProxyTranscriptError,
    build_response_metadata,
    canonical_json,
    client_request_fingerprint,
    derive_proxy_input,
    final_provider_fingerprint,
    idempotency_tool_projection,
    normalize_assistant_tool_calls,
    response_retrieval_text,
)


@dataclass
class _Message:
    role: str
    content: Any = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    model_extra: dict[str, Any] | None = None


def _assistant_calls() -> _Message:
    return _Message(
        role="assistant",
        tool_calls=[
            {
                "id": "call_weather",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": {"city": "Málaga", "days": [1, 2]},
                },
            },
            {
                "id": "call_clock",
                "type": "function",
                "function": {
                    "name": "clock",
                    "arguments": '{"zone":"UTC"}',
                },
            },
        ],
        model_extra={"id": "msg_parent_response"},
    )


def test_trailing_multi_tool_results_preserve_order_types_and_parent_calls() -> None:
    projection = derive_proxy_input(
        [
            _Message(role="user", content="Check both."),
            _assistant_calls(),
            _Message(
                role="tool",
                tool_call_id="call_weather",
                name="weather",
                content={"temperature": 27, "ok": True, "notes": ["dry", 2]},
                model_extra={"status": "success"},
            ),
            _Message(
                role="tool",
                tool_call_id="call_clock",
                content="12:00:00",
            ),
        ]
    )

    assert projection.message_role == "tool"
    assert projection.parent_response_hint == "msg_parent_response"
    results = projection.tool_projection["results"]
    assert [result["tool_call_id"] for result in results] == [
        "call_weather",
        "call_clock",
    ]
    assert results[0]["content"] == {
        "temperature": 27,
        "ok": True,
        "notes": ["dry", 2],
    }
    assert results[0]["extra"] == {"status": "success"}
    assert "untrusted data" in projection.text


@pytest.mark.parametrize(
    "messages",
    [
        [_Message(role="tool", tool_call_id="unknown", content="x")],
        [_assistant_calls(), _Message(role="tool", content="x")],
        [
            _assistant_calls(),
            _Message(role="tool", tool_call_id="unknown", content="x"),
        ],
        [
            _assistant_calls(),
            _Message(role="tool", tool_call_id="call_weather", content="x"),
            _Message(role="user", content="also do something else"),
        ],
    ],
)
def test_ambiguous_or_causally_invalid_tool_suffix_is_rejected(
    messages: list[_Message],
) -> None:
    with pytest.raises(ProxyTranscriptError):
        derive_proxy_input(messages)


def test_new_user_after_completed_tool_cycle_is_not_treated_as_mixed_suffix() -> None:
    projection = derive_proxy_input(
        [
            _Message(role="user", content="Check both."),
            _assistant_calls(),
            _Message(
                role="tool",
                tool_call_id="call_weather",
                content={"temperature": 27},
            ),
            _Message(role="assistant", content="It is 27 degrees."),
            _Message(role="user", content="What about tomorrow?"),
        ]
    )

    assert projection.message_role == "user"
    assert projection.text == "What about tomorrow?"


def test_tool_call_projection_preserves_literal_argument_shapes_and_order() -> None:
    calls = normalize_assistant_tool_calls(_assistant_calls().tool_calls or [])

    assert [call["id"] for call in calls] == ["call_weather", "call_clock"]
    assert calls[0]["arguments"] == {"city": "Málaga", "days": [1, 2]}
    assert calls[1]["arguments"] == '{"zone":"UTC"}'
    assert response_retrieval_text("", calls).startswith("[Assistant tool calls")


def test_tool_metadata_idempotency_sorts_objects_but_preserves_array_order_and_types() -> (
    None
):
    first = {
        "atagia_proxy_transcript": {
            "tool_projection": {
                "schema_version": 1,
                "kind": "assistant_tool_calls",
                "calls": [{"id": "one", "arguments": {"b": 2, "a": 1}}],
            },
            "diagnostic": "ignored",
        }
    }
    same = {
        "atagia_proxy_transcript": {
            "tool_projection": {
                "kind": "assistant_tool_calls",
                "calls": [{"arguments": {"a": 1, "b": 2}, "id": "one"}],
                "schema_version": 1,
            },
            "diagnostic": "different but incidental",
        }
    }
    reordered = {
        "atagia_proxy_transcript": {
            "tool_projection": {
                "schema_version": 1,
                "kind": "assistant_tool_calls",
                "calls": [
                    {"id": "two", "arguments": "1"},
                    {"id": "one", "arguments": 1},
                ],
            }
        }
    }

    assert idempotency_tool_projection(first) == idempotency_tool_projection(same)
    assert idempotency_tool_projection(first) != idempotency_tool_projection(reordered)
    assert canonical_json("1") != canonical_json(1)


def test_response_metadata_replays_tool_only_turn_without_fake_usage() -> None:
    metadata = build_response_metadata(
        pair_id="ptr_1",
        request_message_id="msg_req",
        response_message_id="msg_res",
        client_request_fingerprint="client-fp",
        final_provider_fingerprint="provider-fp",
        content="",
        tool_calls=_assistant_calls().tool_calls or [],
        finish_reason="tool_calls",
        usage=None,
        model="upstream-model",
    )
    replay = metadata["atagia_proxy_transcript"]["replay"]
    stored_calls = metadata["atagia_proxy_transcript"]["tool_projection"]["calls"]

    assert replay["content"] == ""
    assert replay["finish_reason"] == "tool_calls"
    assert replay["usage"] is None
    assert replay["tool_calls"][0]["function"]["arguments"] == (
        '{"city":"Málaga","days":[1,2]}'
    )
    assert stored_calls[0]["arguments"] == {"city": "Málaga", "days": [1, 2]}


def test_client_fingerprint_covers_semantics_but_excludes_stream_and_pair_ids() -> None:
    base = {
        "model": "atagia-memory",
        "messages": [
            {"role": "system", "content": "System A"},
            {"role": "developer", "content": "Developer A"},
            {"role": "user", "content": "Hello"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_completion_tokens": 100,
        "stream": False,
        "metadata": {
            "atagia_message_id": "msg_req_a",
            "atagia_response_message_id": "msg_res_a",
            "response_mode": "normal",
        },
        "vendor_generation_option": {"reasoning": "low"},
    }
    identity = SimpleNamespace(
        user_id="usr_1",
        conversation_id="cnv_1",
        mode="coding_debug",
        cross_chat_memory=False,
        message_id="msg_req_a",
        response_message_id="msg_res_a",
    )
    authority = SimpleNamespace(
        privacy_enforcement="enforce",
        effective_privacy_enforcement="enforce",
        normalized_privilege_level="standard",
        authenticated_user_is_atagia_master=False,
        trusted_evaluation=False,
        authority_source="ordinary_http_boundary:service_api_key",
    )
    fingerprint = client_request_fingerprint(
        base,
        resolved_identity=identity,
        authority=authority,
    )
    presentation_change = {
        **base,
        "stream": True,
        "metadata": {
            **base["metadata"],
            "atagia_message_id": "msg_req_b",
            "atagia_response_message_id": "msg_res_b",
        },
    }
    assert (
        client_request_fingerprint(
            presentation_change,
            resolved_identity=SimpleNamespace(
                **{
                    **identity.__dict__,
                    "message_id": "msg_req_b",
                    "response_message_id": "msg_res_b",
                }
            ),
            authority=authority,
        )
        == fingerprint
    )

    for changed in (
        {**base, "temperature": 0.3},
        {**base, "model": "other-model"},
        {**base, "max_completion_tokens": 101},
        {**base, "vendor_generation_option": {"reasoning": "high"}},
        {
            **base,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        {
            **base,
            "messages": [
                {"role": "developer", "content": "Developer A"},
                {"role": "system", "content": "System A"},
                {"role": "user", "content": "Hello"},
            ],
        },
    ):
        assert (
            client_request_fingerprint(
                changed,
                resolved_identity=identity,
                authority=authority,
            )
            != fingerprint
        )

    changed_identity = SimpleNamespace(
        **{
            **identity.__dict__,
            "mode": "companion",
        }
    )
    assert (
        client_request_fingerprint(
            base,
            resolved_identity=changed_identity,
            authority=authority,
        )
        != fingerprint
    )
    changed_authority = SimpleNamespace(
        **{
            **authority.__dict__,
            "authority_source": "trusted_library",
        }
    )
    assert (
        client_request_fingerprint(
            base,
            resolved_identity=identity,
            authority=changed_authority,
        )
        != fingerprint
    )


def test_final_provider_fingerprint_covers_injected_context_exactly() -> None:
    first = {
        "model": "upstream",
        "messages": [{"role": "system", "content": "Memory snapshot A"}],
        "temperature": 0.2,
    }
    second = {
        **first,
        "messages": [{"role": "system", "content": "Memory snapshot B"}],
    }

    assert final_provider_fingerprint(first) != final_provider_fingerprint(second)
