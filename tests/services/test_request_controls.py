"""Boundary tests for canonical request control resolution."""

from __future__ import annotations

import pytest
from starlette.datastructures import Headers

from atagia.services.request_controls import resolve_memory_scope_controls


@pytest.mark.parametrize("incognito", [None, False, True])
@pytest.mark.parametrize("cross_chat_memory", [None, False, True, "invalid"])
def test_memory_scope_control_matrix(
    incognito: bool | None,
    cross_chat_memory: bool | str | None,
) -> None:
    typed_fields = {}
    if incognito is not None:
        typed_fields["incognito"] = incognito
    if cross_chat_memory is not None:
        typed_fields["cross_chat_memory"] = cross_chat_memory

    if cross_chat_memory == "invalid":
        with pytest.raises(
            ValueError,
            match="Invalid boolean for cross_chat_memory from typed.cross_chat_memory",
        ):
            resolve_memory_scope_controls(typed_fields=typed_fields)
        return

    resolved = resolve_memory_scope_controls(typed_fields=typed_fields)

    assert resolved.incognito is incognito
    assert resolved.cross_chat_memory is (
        False
        if incognito is True
        else True
        if cross_chat_memory is None
        else cross_chat_memory
    )


@pytest.mark.parametrize(
    ("typed_fields", "metadata", "headers", "expected"),
    [
        (
            {"incognito": False},
            {"atagia_incognito": "false"},
            {"X-Atagia-Incognito": "0"},
            (False, True),
        ),
        (
            {"cross_chat_memory": False},
            {"atagia_cross_chat_memory": "off"},
            {"X-Atagia-Cross-Chat-Memory": "no"},
            (None, False),
        ),
        (
            {"incognito": True, "cross_chat_memory": True},
            {"incognito": "true", "cross_chat_memory": "1"},
            {
                "X-Atagia-Incognito": "yes",
                "X-Atagia-Cross-Chat-Memory": "on",
            },
            (True, False),
        ),
    ],
)
def test_memory_scope_consistent_cross_source_duplicates_are_accepted(
    typed_fields: dict[str, object],
    metadata: dict[str, object],
    headers: dict[str, object],
    expected: tuple[bool | None, bool],
) -> None:
    resolved = resolve_memory_scope_controls(
        typed_fields=typed_fields,
        metadata=metadata,
        headers=headers,
    )

    assert (resolved.incognito, resolved.cross_chat_memory) == expected


@pytest.mark.parametrize(
    ("setting", "typed_fields", "metadata", "headers"),
    [
        ("incognito", {"incognito": True}, {"incognito": False}, {}),
        (
            "incognito",
            {"incognito": True},
            {},
            {"X-Atagia-Incognito": "false"},
        ),
        (
            "incognito",
            {},
            {"atagia_incognito": True},
            {"X-Atagia-Incognito": "false"},
        ),
        (
            "cross_chat_memory",
            {"cross_chat_memory": False},
            {"cross_chat_memory": True},
            {},
        ),
        (
            "cross_chat_memory",
            {"cross_chat_memory": False},
            {},
            {"X-Atagia-Cross-Chat-Memory": "true"},
        ),
        (
            "cross_chat_memory",
            {},
            {"atagia_cross_chat_memory": False},
            {"X-Atagia-Cross-Chat-Memory": "true"},
        ),
    ],
)
def test_memory_scope_cross_source_contradictions_are_rejected(
    setting: str,
    typed_fields: dict[str, object],
    metadata: dict[str, object],
    headers: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match=f"Conflicting {setting} values across request sources",
    ):
        resolve_memory_scope_controls(
            typed_fields=typed_fields,
            metadata=metadata,
            headers=headers,
        )


@pytest.mark.parametrize(
    ("metadata", "headers", "message"),
    [
        (
            {"incognito": "maybe"},
            {},
            "Invalid boolean for incognito from metadata.incognito",
        ),
        (
            {},
            {"X-Atagia-Cross-Chat-Memory": "sometimes"},
            "Invalid boolean for cross_chat_memory from header.X-Atagia-Cross-Chat-Memory",
        ),
    ],
)
def test_memory_scope_invalid_cross_source_boolean_is_rejected(
    metadata: dict[str, object],
    headers: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_memory_scope_controls(metadata=metadata, headers=headers)


def test_memory_scope_repeated_identical_headers_are_all_accepted() -> None:
    headers = Headers(
        raw=[
            (b"x-atagia-cross-chat-memory", b"false"),
            (b"x-atagia-cross-chat-memory", b"off"),
        ]
    )

    resolved = resolve_memory_scope_controls(headers=headers)

    assert resolved.cross_chat_memory is False


@pytest.mark.parametrize(
    "header_name",
    ["x-atagia-incognito", "x-atagia-cross-chat-memory"],
)
@pytest.mark.parametrize(
    "values",
    [(b"false", b"true"), (b"true", b"false")],
)
def test_memory_scope_repeated_contradictory_headers_are_rejected(
    header_name: str,
    values: tuple[bytes, bytes],
) -> None:
    headers = Headers(
        raw=[
            (header_name.encode(), values[0]),
            (header_name.encode(), values[1]),
        ]
    )
    setting = header_name.removeprefix("x-atagia-").replace("-", "_")

    with pytest.raises(
        ValueError,
        match=f"Conflicting {setting} values across request sources",
    ):
        resolve_memory_scope_controls(headers=headers)
