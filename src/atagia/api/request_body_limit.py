"""ASGI middleware enforcing encoded HTTP request-body limits."""

from __future__ import annotations

import json
from typing import Any

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse


class RequestBodyLimitExceeded(HTTPException):
    """Signal a streamed body overflow without being rewritten as a parse error."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        super().__init__(
            status_code=413,
            detail=f"Request body exceeds the configured limit of {self.limit} bytes",
        )


def request_body_limit_error_response(path: str, limit: int) -> JSONResponse:
    """Return the route-appropriate stable response for an encoded-body overflow."""

    return JSONResponse(
        status_code=413,
        content=_too_large_payload(path, limit),
    )


class RequestBodyLimitMiddleware:
    """Reject Content-Length and chunked bodies once they cross the cap."""

    def __init__(self, app: Any, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = int(max_body_bytes)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope.get("headers", ()))
        if content_length is not None and content_length > self.max_body_bytes:
            await _send_too_large(scope, send, self.max_body_bytes)
            return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise RequestBodyLimitExceeded(self.max_body_bytes)
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyLimitExceeded:
            if response_started:
                raise
            await _send_too_large(scope, send, self.max_body_bytes)


def _content_length(headers: Any) -> int | None:
    for raw_name, raw_value in headers:
        if bytes(raw_name).lower() != b"content-length":
            continue
        try:
            value = int(bytes(raw_value).decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            return None
        return value if value >= 0 else None
    return None


async def _send_too_large(
    scope: dict[str, Any],
    send: Any,
    limit: int,
) -> None:
    path = str(scope.get("path") or "")
    payload = _too_large_payload(path, limit)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _too_large_payload(path: str, limit: int) -> dict[str, Any]:
    message = f"Request body exceeds the configured limit of {limit} bytes"
    if path == "/v1/chat/completions":
        return {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": "request_body_too_large",
            }
        }
    return {"detail": message}
