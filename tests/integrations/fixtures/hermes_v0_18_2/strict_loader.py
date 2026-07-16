"""Strict register(ctx) subset of the pinned Hermes memory plugin loader."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from agent.memory_provider import MemoryProvider


class ProviderCollector:
    def __init__(self) -> None:
        self.provider: MemoryProvider | None = None

    def register_memory_provider(self, provider: MemoryProvider) -> None:
        if not isinstance(provider, MemoryProvider):
            raise TypeError("registered provider does not implement MemoryProvider")
        self.provider = provider


def load_registered_provider(provider_dir: Path) -> MemoryProvider:
    """Load the package and require the official register entrypoint.

    Hermes itself has a permissive subclass-scan fallback. This gate
    intentionally does not use it, so a missing/broken register(ctx) cannot
    make Atagia's contract CI pass.
    """
    module_name = f"_hermes_contract_memory_{provider_dir.name}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        provider_dir / "__init__.py",
        submodule_search_locations=[str(provider_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("provider package is not loadable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        register = getattr(module, "register", None)
        if not callable(register):
            raise TypeError("provider must expose register(ctx)")
        collector = ProviderCollector()
        register(collector)
        if collector.provider is None:
            raise TypeError("register(ctx) did not register a memory provider")
        return collector.provider
    except Exception:
        _remove_package(module_name)
        raise


def unload_provider(provider: MemoryProvider) -> None:
    provider.shutdown()
    package = provider.__class__.__module__.split(".provider", 1)[0]
    _remove_package(package)


def _remove_package(package: str) -> None:
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            sys.modules.pop(name, None)
