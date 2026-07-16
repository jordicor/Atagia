"""Faithful lifecycle driver for the pinned Atagia Hermes downstream patch.

It models the patch's public provider calls with durable SessionDB-style row
IDs. The production patch is shipped under integrations/hermes/patches/.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "hermes.memory-selection.v1"
HERMES_VERSION = "0.18.2"
HERMES_COMMIT = "e4ea0a0ed7fc24761b2b425146893561a73216e1"


class PatchedMemorySelectionHost:
    def __init__(
        self,
        provider: Any,
        *,
        session_id: str,
        hermes_home: Path,
        selected_messages: list[dict[str, Any]] | None = None,
        next_row_id: int = 1,
        next_turn_number: int = 1,
    ) -> None:
        self.provider = provider
        self.session_id = session_id
        self.selected_messages = deepcopy(selected_messages or [])
        self.next_row_id = next_row_id
        self.next_turn_number = next_turn_number
        self.current_user: dict[str, Any] | None = None
        provider.initialize(
            session_id,
            hermes_home=str(hermes_home),
            platform="cli",
            agent_context="primary",
            host_capabilities={
                CONTRACT_VERSION: {
                    "contract_version": CONTRACT_VERSION,
                    "hermes_version": HERMES_VERSION,
                    "hermes_commit": HERMES_COMMIT,
                }
            },
        )

    def start_turn(
        self,
        text: str,
        *,
        mutation_kind: str = "append",
        prefetch: bool = True,
        row_id: int | None = None,
    ) -> str:
        if self.current_user is not None:
            raise RuntimeError("the prior host turn is still open")
        current = self._message("user", text, row_id=row_id)
        self.current_user = current
        signal = self._signal(mutation_kind, current_user=current)
        self.provider.on_turn_start(
            self.next_turn_number,
            text,
            session_id=self.session_id,
            memory_selection=signal,
        )
        self.next_turn_number += 1
        return (
            self.provider.prefetch(text, session_id=self.session_id) if prefetch else ""
        )

    def complete_turn(
        self,
        text: str,
        *,
        row_id: int | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if self.current_user is None:
            raise RuntimeError("no host turn is open")
        assistant = self._message("assistant", text, row_id=row_id)
        selected = (
            deepcopy(messages)
            if messages is not None
            else [*deepcopy(self.selected_messages), self.current_user, assistant]
        )
        self.provider.sync_turn(
            str(self.current_user["content"]),
            text,
            session_id=self.session_id,
            messages=selected,
        )
        self.selected_messages = deepcopy(selected)
        self.current_user = None
        return selected

    def retry_last(
        self,
        text: str,
        *,
        mutation_kind: str = "retry",
    ) -> str:
        self._truncate_turns(1)
        return self.start_turn(text, mutation_kind=mutation_kind)

    def undo(self, turns: int = 1) -> None:
        self._truncate_turns(turns)
        self.provider.on_session_switch(
            self.session_id,
            rewound=True,
            memory_selection=self._signal("undo", current_user=None),
        )

    def end_session(self) -> None:
        self.provider.on_session_end(deepcopy(self.selected_messages))

    def _truncate_turns(self, turns: int) -> None:
        remove = turns * 2
        if remove < 1 or remove > len(self.selected_messages):
            raise ValueError("cannot truncate that many selected turns")
        self.selected_messages = self.selected_messages[:-remove]

    def _signal(
        self,
        mutation_kind: str,
        *,
        current_user: dict[str, Any] | None,
    ) -> dict[str, Any]:
        cutoff = (
            self.selected_messages[-1]["host_message_id"]
            if self.selected_messages
            else None
        )
        return {
            "contract_version": CONTRACT_VERSION,
            "session_id": self.session_id,
            "mutation_kind": mutation_kind,
            "retained_cutoff_host_message_id": cutoff,
            "messages": deepcopy(self.selected_messages),
            "current_user_message": deepcopy(current_user),
        }

    def _message(
        self,
        role: str,
        text: str,
        *,
        row_id: int | None,
    ) -> dict[str, Any]:
        resolved = self.next_row_id if row_id is None else row_id
        self.next_row_id = max(self.next_row_id, resolved + 1)
        return {
            "role": role,
            "content": text,
            "host_message_id": f"hermes-sqlite-message:{resolved}",
            "generation_id": "default",
        }
