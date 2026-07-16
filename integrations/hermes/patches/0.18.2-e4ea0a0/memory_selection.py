"""Versioned selected-transcript signals for mutation-safe memory providers."""

from __future__ import annotations

from typing import Any


CONTRACT_VERSION = "hermes.memory-selection.v1"
HERMES_VERSION = "0.18.2"
HERMES_COMMIT = "e4ea0a0ed7fc24761b2b425146893561a73216e1"


def host_capabilities() -> dict[str, dict[str, str]]:
    return {
        CONTRACT_VERSION: {
            "contract_version": CONTRACT_VERSION,
            "hermes_version": HERMES_VERSION,
            "hermes_commit": HERMES_COMMIT,
        }
    }


def mark_next_turn_mutation(agent: Any, mutation_kind: str) -> None:
    if mutation_kind not in {"retry", "regeneration"}:
        raise ValueError("unsupported next-turn memory mutation")
    agent._memory_selection_next_mutation = mutation_kind


def build_turn_signal(agent: Any) -> dict[str, Any]:
    rows = _active_rows(agent)
    if not rows or str(rows[-1].get("role") or "") != "user":
        raise RuntimeError("the durable current user row is unavailable")
    current = _project(rows[-1])
    selected = _selected_messages(rows[:-1])
    mutation_kind = getattr(agent, "_memory_selection_next_mutation", "append")
    agent._memory_selection_next_mutation = "append"
    return _signal(
        agent,
        mutation_kind=mutation_kind,
        selected=selected,
        current_user=current,
    )


def build_rewind_signal(agent: Any, mutation_kind: str = "undo") -> dict[str, Any]:
    if mutation_kind not in {"undo", "retry", "regeneration"}:
        raise ValueError("unsupported rewind memory mutation")
    return _signal(
        agent,
        mutation_kind=mutation_kind,
        selected=_selected_messages(_active_rows(agent)),
        current_user=None,
    )


def selected_transcript(agent: Any) -> list[dict[str, Any]]:
    rows = _active_rows(agent)
    try:
        return _selected_messages(rows)
    except RuntimeError:
        # Deliver an invalid terminal boundary to the provider instead of
        # silently dropping it. The provider owns the fail-closed decision.
        return [
            _project(row, allow_empty=True)
            for row in rows
            if str(row.get("role") or "").lower() in {"user", "assistant"}
        ]


def _signal(
    agent: Any,
    *,
    mutation_kind: str,
    selected: list[dict[str, Any]],
    current_user: dict[str, Any] | None,
) -> dict[str, Any]:
    cutoff = selected[-1]["host_message_id"] if selected else None
    return {
        "contract_version": CONTRACT_VERSION,
        "session_id": str(agent.session_id or ""),
        "mutation_kind": mutation_kind,
        "retained_cutoff_host_message_id": cutoff,
        "messages": selected,
        "current_user_message": current_user,
    }


def _active_rows(agent: Any) -> list[dict[str, Any]]:
    database = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    if database is None or not session_id:
        raise RuntimeError("Hermes SessionDB is required for memory selection")
    return list(database.get_messages(session_id, include_inactive=False))


def _selected_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    pending_user: dict[str, Any] | None = None
    final_assistant: dict[str, Any] | None = None
    continuation_user_expected = False
    tool_chain_open = False
    for raw in rows:
        role = str(raw.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        if role == "user":
            projected = _project(raw)
            if pending_user is not None and tool_chain_open:
                raise RuntimeError("durable transcript ends inside a tool chain")
            if pending_user is not None and continuation_user_expected:
                final_assistant = None
                continuation_user_expected = False
                continue
            if pending_user is not None and final_assistant is not None:
                selected.extend((pending_user, final_assistant))
            pending_user = projected
            final_assistant = None
            continuation_user_expected = False
            tool_chain_open = False
        elif pending_user is not None:
            if raw.get("tool_calls"):
                final_assistant = None
                continuation_user_expected = False
                tool_chain_open = True
                continue
            projected = _project(raw)
            final_assistant = projected
            continuation_user_expected = str(
                raw.get("finish_reason") or ""
            ).strip().lower() in {"incomplete", "length"}
            tool_chain_open = bool(raw.get("tool_calls"))
    if (
        pending_user is not None
        and final_assistant is not None
        and not continuation_user_expected
        and not tool_chain_open
    ):
        selected.extend((pending_user, final_assistant))
    elif pending_user is not None:
        raise RuntimeError("durable transcript does not contain complete turns")
    return selected


def _project(
    row: dict[str, Any],
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    row_id = row.get("id")
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 1:
        raise RuntimeError("durable Hermes message row ID is unavailable")
    content = row.get("content")
    if not isinstance(content, str) or (not content.strip() and not allow_empty):
        raise RuntimeError("durable Hermes message text is unavailable")
    return {
        "role": str(row.get("role") or "").lower(),
        "content": content,
        "host_message_id": f"hermes-sqlite-message:{row_id}",
        "generation_id": "default",
        "finish_reason": row.get("finish_reason"),
        "tool_calls": row.get("tool_calls"),
    }
