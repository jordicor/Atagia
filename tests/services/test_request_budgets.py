"""Boundary tests for external request and attachment budgets."""

from __future__ import annotations

import base64
from types import SimpleNamespace
import tracemalloc

import pytest

from atagia.services.request_budgets import (
    DEFAULT_REQUEST_MAX_ATTACHMENT_DECODED_BYTES,
    DEFAULT_REQUEST_MAX_ATTACHMENTS,
    DEFAULT_REQUEST_MAX_ATTACHMENTS_DECODED_BYTES,
    DEFAULT_REQUEST_MAX_BODY_BYTES,
    DEFAULT_REQUEST_MAX_MESSAGE_TEXT_BYTES,
    DEFAULT_REQUEST_MAX_METADATA_BYTES,
    RequestBudgetExceededError,
    RequestBudgetLimits,
    RequestPayloadStructureError,
    decoded_base64_size,
    validate_direct_message_request_budget,
    validate_openai_proxy_request_budget,
)


def _limits(**overrides: int) -> RequestBudgetLimits:
    values = {
        "body_bytes": 1_024,
        "message_text_bytes": 8,
        "attachments": 2,
        "attachment_decoded_bytes": 6,
        "attachments_decoded_bytes": 9,
        "metadata_bytes": 12,
    }
    values.update(overrides)
    return RequestBudgetLimits(**values)


def _proxy_request(*contents: object, metadata: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        messages=[SimpleNamespace(content=content) for content in contents],
        metadata=metadata,
    )


def test_request_budget_defaults_match_the_external_contract() -> None:
    limits = RequestBudgetLimits.from_settings(SimpleNamespace())

    assert limits == RequestBudgetLimits(
        body_bytes=DEFAULT_REQUEST_MAX_BODY_BYTES,
        message_text_bytes=DEFAULT_REQUEST_MAX_MESSAGE_TEXT_BYTES,
        attachments=DEFAULT_REQUEST_MAX_ATTACHMENTS,
        attachment_decoded_bytes=DEFAULT_REQUEST_MAX_ATTACHMENT_DECODED_BYTES,
        attachments_decoded_bytes=DEFAULT_REQUEST_MAX_ATTACHMENTS_DECODED_BYTES,
        metadata_bytes=DEFAULT_REQUEST_MAX_METADATA_BYTES,
    )


@pytest.mark.parametrize("value", ["1234567", "12345678", "éééé"])
def test_direct_text_accepts_values_at_or_below_the_utf8_byte_limit(value: str) -> None:
    validate_direct_message_request_budget(
        message_text=value,
        attachments=[],
        metadata=None,
        limits=_limits(),
    )


def test_direct_text_rejects_one_byte_above_the_utf8_limit() -> None:
    with pytest.raises(RequestBudgetExceededError, match="message_text"):
        validate_direct_message_request_budget(
            message_text="123456789",
            attachments=[],
            metadata=None,
            limits=_limits(),
        )


def test_direct_attachment_count_has_an_exact_boundary() -> None:
    attachment = {"content_text": "a"}
    validate_direct_message_request_budget(
        message_text="ok",
        attachments=[attachment, attachment],
        metadata=None,
        limits=_limits(),
    )

    with pytest.raises(RequestBudgetExceededError, match="attachments"):
        validate_direct_message_request_budget(
            message_text="ok",
            attachments=[attachment, attachment, attachment],
            metadata=None,
            limits=_limits(),
        )


def test_decoded_blob_and_aggregate_limits_are_independent() -> None:
    six_bytes = base64.b64encode(b"123456").decode("ascii")
    validate_direct_message_request_budget(
        message_text="ok",
        attachments=[{"content_base64": six_bytes}],
        metadata=None,
        limits=_limits(),
    )

    seven_bytes = base64.b64encode(b"1234567").decode("ascii")
    with pytest.raises(
        RequestBudgetExceededError, match=r"attachments\.0\.decoded_bytes"
    ):
        validate_direct_message_request_budget(
            message_text="ok",
            attachments=[{"content_base64": seven_bytes}],
            metadata=None,
            limits=_limits(),
        )

    five_bytes = base64.b64encode(b"12345").decode("ascii")
    with pytest.raises(RequestBudgetExceededError, match="attachments.decoded_bytes"):
        validate_direct_message_request_budget(
            message_text="ok",
            attachments=[
                {"content_base64": five_bytes},
                {"content_base64": five_bytes},
            ],
            metadata=None,
            limits=_limits(),
        )


def test_metadata_serialized_byte_limit_accepts_exact_and_rejects_above() -> None:
    metadata = {"key": "ñ"}
    serialized_bytes = len('{"key":"ñ"}'.encode("utf-8"))

    validate_direct_message_request_budget(
        message_text="ok",
        attachments=[],
        metadata=metadata,
        limits=_limits(metadata_bytes=serialized_bytes),
    )
    with pytest.raises(RequestBudgetExceededError, match="metadata"):
        validate_direct_message_request_budget(
            message_text="ok",
            attachments=[],
            metadata=metadata,
            limits=_limits(metadata_bytes=serialized_bytes - 1),
        )


@pytest.mark.parametrize(
    "value",
    ["A", "A===", "AA=A", "AA?=", "====", "data that is not base64"],
)
def test_invalid_base64_is_a_structural_error(value: str) -> None:
    with pytest.raises(RequestPayloadStructureError, match="invalid base64"):
        decoded_base64_size(value, field="payload")


def test_openai_multimodal_budgets_cover_data_urls_text_count_and_metadata() -> None:
    six_bytes = base64.b64encode(b"123456").decode("ascii")
    request = _proxy_request(
        "12345678",
        [
            {"type": "text", "text": "12345678"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{six_bytes}"},
            },
        ],
        metadata={"a": "b"},
    )
    validate_openai_proxy_request_budget(request, limits=_limits(metadata_bytes=20))

    too_many = _proxy_request(
        [
            {"type": "image_url", "image_url": {"url": "https://example/a"}},
            {"type": "input_audio", "input_audio": {"data": ""}},
            {"type": "document_url", "url": "https://example/b"},
        ]
    )
    with pytest.raises(RequestBudgetExceededError, match="attachments"):
        validate_openai_proxy_request_budget(too_many, limits=_limits())


def test_openai_expansive_base64_is_rejected_without_decoding_it() -> None:
    encoded = base64.b64encode(b"1234567").decode("ascii")
    request = _proxy_request(
        [{"type": "input_file", "data": encoded}],
    )

    with pytest.raises(RequestBudgetExceededError, match="decoded_bytes"):
        validate_openai_proxy_request_budget(request, limits=_limits())


@pytest.mark.parametrize(
    "content",
    [
        {"type": "text", "text": "123456789"},
        {"content": "123456789"},
        {"message": "123456789"},
        ["1234", "5678"],
        {
            "multi_ai": True,
            "responses": [{"model": "m", "content": "123456789"}],
        },
    ],
    ids=(
        "single-text-block",
        "content-wrapper",
        "message-wrapper",
        "raw-string-list",
        "multi-ai-response",
    ),
)
def test_openai_text_budget_covers_every_projected_content_shape(
    content: object,
) -> None:
    with pytest.raises(RequestBudgetExceededError, match=r"messages\.0\.content"):
        validate_openai_proxy_request_budget(_proxy_request(content), limits=_limits())


@pytest.mark.parametrize(
    ("content", "expected_field"),
    [
        (
            {"type": "input_file", "data": "MTIzNDU2Nw=="},
            "messages.0.content.decoded_bytes",
        ),
        (
            {"content": {"type": "input_file", "data": "MTIzNDU2Nw=="}},
            "messages.0.content.content.decoded_bytes",
        ),
        (
            {"message": [{"type": "input_file", "data": "MTIzNDU2Nw=="}]},
            "messages.0.content.message.0.decoded_bytes",
        ),
        (
            {
                "multi_ai": True,
                "responses": [
                    {
                        "model": "m",
                        "content": {"type": "input_file", "data": "MTIzNDU2Nw=="},
                    }
                ],
            },
            "messages.0.content.responses.0.content.decoded_bytes",
        ),
    ],
    ids=("single-block", "content-wrapper", "message-wrapper", "multi-ai-response"),
)
def test_openai_attachment_budget_covers_nested_projected_shapes(
    content: object,
    expected_field: str,
) -> None:
    with pytest.raises(RequestBudgetExceededError) as exc_info:
        validate_openai_proxy_request_budget(_proxy_request(content), limits=_limits())

    assert exc_info.value.field == expected_field


def test_openai_attachment_metadata_uses_the_metadata_byte_cap() -> None:
    request = _proxy_request(
        {
            "type": "input_file",
            "data": "",
            "metadata": {"key": "12345"},
        }
    )

    with pytest.raises(RequestBudgetExceededError) as exc_info:
        validate_openai_proxy_request_budget(request, limits=_limits())

    assert exc_info.value.field == "messages.0.content.metadata"


def test_openai_attachment_count_rejects_before_text_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def projection_must_not_start(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("text projection started before attachment rejection")

    monkeypatch.setattr(
        "atagia.services.request_budgets.iter_message_text_chunks",
        projection_must_not_start,
    )
    request = _proxy_request(
        [{"type": "input_image", "image_url": {"url": "https://example"}}] * 3
    )

    with pytest.raises(RequestBudgetExceededError) as exc_info:
        validate_openai_proxy_request_budget(request, limits=_limits())

    assert exc_info.value.field == "attachments"


def test_openai_large_tool_string_rejects_before_json_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def encoder_must_not_start(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("JSON encoding started before the string lower bound")

    monkeypatch.setattr(
        "atagia.services.request_budgets.iter_tool_message_text_chunks",
        encoder_must_not_start,
    )
    request = SimpleNamespace(
        messages=[
            SimpleNamespace(
                role="tool",
                content={"result": "x" * 1_000_000},
            )
        ],
        metadata=None,
    )

    with pytest.raises(RequestBudgetExceededError) as exc_info:
        validate_openai_proxy_request_budget(request, limits=_limits())

    assert exc_info.value.field == "messages.0.content"


def test_openai_deep_content_nesting_is_a_stable_structural_error() -> None:
    content: object = "ok"
    for _ in range(1_100):
        content = {"content": content}

    with pytest.raises(RequestPayloadStructureError) as exc_info:
        validate_openai_proxy_request_budget(
            _proxy_request(content),
            limits=_limits(message_text_bytes=32_000),
        )

    assert exc_info.value.field == "messages.0.content"
    assert "unsupported structure" in exc_info.value.message


def test_large_external_values_are_rejected_without_full_size_temporary_copies() -> (
    None
):
    large_text = "x" * (4 * 1024 * 1024)
    large_data_url = "data:image/png;base64," + ("A" * (4 * 1024 * 1024))
    cases = {
        "direct-message": lambda: validate_direct_message_request_budget(
            message_text=large_text,
            attachments=[],
            metadata=None,
            limits=_limits(),
        ),
        "direct-content-text": lambda: validate_direct_message_request_budget(
            message_text="ok",
            attachments=[{"content_text": large_text}],
            metadata=None,
            limits=_limits(),
        ),
        "metadata": lambda: validate_direct_message_request_budget(
            message_text="ok",
            attachments=[],
            metadata={"value": large_text},
            limits=_limits(),
        ),
        "inline-data-url": lambda: validate_openai_proxy_request_budget(
            _proxy_request(
                [{"type": "input_image", "image_url": {"url": large_data_url}}]
            ),
            limits=_limits(),
        ),
    }

    for case_name, validate in cases.items():
        tracemalloc.start()
        try:
            try:
                validate()
            except RequestBudgetExceededError:
                pass
            else:
                raise AssertionError(f"{case_name} was not rejected")
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 1024 * 1024, f"{case_name} allocated {peak} traced bytes"


@pytest.mark.parametrize(
    "data_url",
    [
        "DATA:image/png;base64,MTIzNDU2Nw==",
        "DaTa:image/png;BASE64,MTIzNDU2Nw==",
        " data:image/png;base64,MTIzNDU2Nw==",
        "\tdata:image/png;base64,MTIzNDU2Nw==",
        "\x00data:image/png;base64,MTIzNDU2Nw==",
        "data:image/png;base64,MTIzNDU2Nw==\t ",
        "data:İİİİ;base64,MTIzNDU2Nw==",
    ],
)
def test_openai_data_url_scheme_is_case_insensitive_for_decoded_budgets(
    data_url: str,
) -> None:
    request = _proxy_request(
        [
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        ]
    )

    with pytest.raises(RequestBudgetExceededError, match="decoded_bytes"):
        validate_openai_proxy_request_budget(request, limits=_limits())


def test_openai_remote_url_is_not_misclassified_as_inline_data() -> None:
    request = _proxy_request(
        [
            {
                "type": "image_url",
                "image_url": {"url": " HTTPS://example.invalid/large-image.png "},
            }
        ]
    )

    validate_openai_proxy_request_budget(
        request,
        limits=_limits(attachment_decoded_bytes=1, attachments_decoded_bytes=1),
    )


def test_large_malformed_data_url_fails_structurally_without_payload_scan() -> None:
    request = _proxy_request(
        [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;" + ("x" * 1_000_000)},
            }
        ]
    )

    with pytest.raises(RequestPayloadStructureError, match="base64 data URL"):
        validate_openai_proxy_request_budget(request, limits=_limits())


def test_non_json_metadata_is_a_structural_error() -> None:
    with pytest.raises(RequestPayloadStructureError, match="JSON serializable"):
        validate_direct_message_request_budget(
            message_text="ok",
            attachments=[],
            metadata={"bad": object()},
            limits=_limits(metadata_bytes=1_000),
        )
