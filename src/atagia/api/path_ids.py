"""Central path-parameter decoding for reversible transport identifiers."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import Response
from fastapi.routing import APIRoute

from atagia.transport_ids import decode_path_id


class TransportIdRoute(APIRoute):
    """Decode every dynamic path segment once before dependency resolution."""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original_handler = super().get_route_handler()

        async def decoded_route_handler(request: Request) -> Response:
            try:
                request.scope["path_params"] = {
                    name: decode_path_id(value) if isinstance(value, str) else value
                    for name, value in request.path_params.items()
                }
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
            return await original_handler(request)

        return decoded_route_handler
