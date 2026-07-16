"""Faithful Hermes Agent 0.18.2 MemoryProvider ABI snapshot.

Pinned commit: e4ea0a0ed7fc24761b2b425146893561a73216e1
Official path: agent/memory_provider.py

Only method bodies/docstrings are shortened; abstractness, names, signatures,
defaults, and return contracts match the pinned source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        raise NotImplementedError

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        return None

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError

    def shutdown(self) -> None:
        return None

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        return None

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        return None

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        return None

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        return None

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None

    def backup_paths(self) -> List[str]:
        return []
