"""Atagia memory plugin entrypoint for Hermes Agent 0.18.2."""

from .provider import (
    HERMES_MEMORY_SELECTION_CAPABILITY,
    AtagiaMemoryProvider,
    validate_supported_hermes_host,
)


def register(ctx) -> None:
    """Register through Hermes' official memory-provider collector."""
    validate_supported_hermes_host()
    ctx.register_memory_provider(AtagiaMemoryProvider())


__all__ = [
    "AtagiaMemoryProvider",
    "HERMES_MEMORY_SELECTION_CAPABILITY",
    "register",
]
