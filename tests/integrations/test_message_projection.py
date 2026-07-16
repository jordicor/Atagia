from __future__ import annotations

import json
import random
import string

from atagia.integrations.message_projection import message_to_text


def test_message_to_text_projects_multimodal_blocks_without_binary_payloads() -> None:
    message = [
        {"type": "text", "text": "remember this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAABBBB"}},
        {"type": "document_bytes", "filename": "brief.pdf", "data": "BASE64PDF"},
        {"type": "audio", "filename": "clip.wav", "data": "BASE64AUDIO"},
    ]

    text = message_to_text(message)

    assert "remember this" in text
    assert "[Image attached]" in text
    assert "[Document attached: brief.pdf]" in text
    assert "[Audio attached: clip.wav]" in text
    assert "AAAABBBB" not in text
    assert "BASE64PDF" not in text
    assert "BASE64AUDIO" not in text


def test_message_to_text_preserves_multi_ai_json_strings_literally() -> None:
    payload = json.dumps(
        {
            "multi_ai": True,
            "responses": [
                {"model": "gpt-5", "content": "First answer"},
                {"machine": "claude", "content": [{"type": "text", "text": "Second"}]},
            ],
        }
    )

    assert message_to_text(payload) == payload


def test_message_to_text_projects_typed_multi_ai_content() -> None:
    payload = {
        "multi_ai": True,
        "responses": [
            {"model": "gpt-5", "content": "First answer"},
            {"machine": "claude", "content": [{"type": "text", "text": "Second"}]},
        ],
    }

    assert message_to_text(payload) == "[gpt-5]\nFirst answer\n\n[claude]\nSecond"


def test_message_to_text_preserves_arbitrary_strings_byte_for_byte() -> None:
    explicit_cases = [
        "",
        "   ",
        "\t\n  {\"key\": [1, 2]}  \r\n",
        '["array", {"nested": "\\\\n"}]',
        "{malformed JSON",
        "```json\n{\"code\": true}\n```",
        "emoji 🧠 and Unicode 日本語 مرحبا",
        "null",
        "true",
        "42",
    ]
    rng = random.Random(20260712)
    alphabet = string.printable + "🧠日本語مرحباñλ"
    generated_cases = [
        "".join(rng.choice(alphabet) for _ in range(length))
        for length in range(0, 257, 7)
    ]

    for value in [*explicit_cases, *generated_cases]:
        assert message_to_text(value) == value


def test_message_to_text_uses_text_file_loader_when_available() -> None:
    block = {
        "type": "text_file",
        "text_file": {"filename": "notes.txt"},
    }

    assert (
        message_to_text(block, text_file_loader=lambda _block: "loaded notes")
        == "loaded notes"
    )


def test_message_to_text_falls_back_to_text_file_placeholder() -> None:
    block = {
        "type": "text_file",
        "text_file": {"filename": "notes.txt"},
    }

    assert message_to_text(block) == "[Text file attached: notes.txt]"
