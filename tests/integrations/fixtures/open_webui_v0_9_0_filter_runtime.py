"""Faithful filter-call subset from Open WebUI v0.9.0.

Pinned commit: f31768e20e5c6b4f6da0ef657877298b359936cf
Official file: backend/open_webui/utils/filter.py
Official file SHA-256: 61e08fb796c042b295e0c6c570429f458c375bc2dec5e9a55e2901b6ed1c1e12

The production function discovers inlet/outlet, inspects its signature, and
passes only supported ``__...__`` parameters plus ``body``. This fixture keeps
that behavior without copying unrelated Open WebUI application dependencies.
"""

from __future__ import annotations

import inspect
from typing import Any


async def call_filter_hook(
    filter_instance: Any,
    hook_name: str,
    body: dict[str, Any],
    *,
    user: dict[str, Any] | None,
    metadata: dict[str, Any],
    event_emitter=None,
) -> dict[str, Any]:
    handler = getattr(filter_instance, hook_name)
    signature = inspect.signature(handler)
    extra_params = {
        "__user__": user,
        "__metadata__": metadata,
        "__event_emitter__": event_emitter,
        "__id__": "atagia-memory-filter",
    }
    kwargs = {
        name: value
        for name, value in extra_params.items()
        if name in signature.parameters
    }
    result = handler(body, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result
