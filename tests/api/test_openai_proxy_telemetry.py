"""A proxy turn is persisted with the same telemetry as a chat turn (CS-1.5).

The OpenAI-compatible proxy runs a complete turn without going through
ChatService, so before CS-1.5 it produced messages, memories, and jobs but no
measured turn at all. These tests pin both shapes: the non-streaming turn, and
the streamed turn whose reply round-trip is only recorded once the stream is
drained, after the setup coroutine has already returned.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
import pytest

from atagia.app import create_app
from atagia.core.db_sqlite import open_connection
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.models.schemas_memory import TurnSurface
from atagia.services.llm_client import LLMClient
from atagia.services.openai_proxy_service import (
    _PROXY_STREAM_ERROR_TYPES,
    OpenAIProxyService,
    ProxyContextAttempt,
    ProxyStreamErrorCode,
)
from atagia.services.sidecar_service import SidecarService

from tests.api.test_openai_proxy import (
    ProxyProvider,
    _ensure_capture_namespace,
    _settings,
)

_HEADERS = {
    "Authorization": "Bearer service-key",
    "X-Atagia-User-Id": "usr_proxy_telemetry",
    "X-Atagia-Platform-Id": "proxy_desktop",
    "X-Atagia-Conversation-Id": "cnv_proxy_telemetry",
}


async def _turn_rows(database_path: Path) -> list[dict[str, Any]]:
    connection = await open_connection(str(database_path))
    connection.row_factory = aiosqlite.Row
    try:
        cursor = await connection.execute(
            """
            SELECT *
            FROM retrieval_events
            WHERE user_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            ("usr_proxy_telemetry",),
        )
        return [dict(row) for row in await cursor.fetchall()]
    finally:
        await connection.close()


def _assert_completed_turn(row: dict[str, Any], *, surface: TurnSurface) -> None:
    assert row["turn_surface"] == surface.value
    assert row["response_message_id"] is not None
    assert row["turn_to_event_write_wall_ms"] > 0.0
    assert row["retrieval_duration_ms"] > 0.0
    # Retrieval is a slice of the turn, so it can never be the longer of the two.
    assert row["retrieval_duration_ms"] <= row["turn_to_event_write_wall_ms"]
    assert row["llm_total_calls"] >= 1
    assert row["llm_failed_calls"] == 0
    # A call was counted above, so its latency cannot be zero: a >= 0.0 check
    # here would only restate the column's CHECK constraint.
    assert row["llm_total_latency_ms"] > 0.0
    # CS-1.4: the trace must carry the count too. The proxy surfaces inherit a
    # row written by the sidecar retrieval, and completing the turn used to
    # leave outcome_json untouched -- so the persisted trace read as "no calls"
    # on exactly the rows where the reply round-trip did happen.
    # This helper reads raw rows, so outcome_json is still encoded text here.
    outcome = json.loads(row["outcome_json"])
    metrics = outcome["llm_call_metrics"]
    assert metrics is not None
    assert metrics["total_calls"] == row["llm_total_calls"]
    assert metrics["failed_calls"] == row["llm_failed_calls"]
    assert "chat_reply" in metrics["by_purpose"]
    trace = outcome.get("retrieval_trace")
    if isinstance(trace, dict):
        # The nested retrieval trace is refreshed from the same object, so a
        # reader cannot find two different counts in one row.
        assert trace["llm_call_metrics"] == metrics


@pytest.mark.asyncio
async def test_proxy_completion_turn_persists_turn_telemetry(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_HEADERS,
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [
                        {"role": "user", "content": "Remember the proxy path."}
                    ],
                },
            )

    assert response.status_code == 200
    rows = await _turn_rows(Path(settings.sqlite_path))
    assert len(rows) == 1
    row = rows[0]
    _assert_completed_turn(row, surface=TurnSurface.PROXY_COMPLETION)
    # The reply is the proxy's own provider call and must be inside the count.
    assert row["llm_by_purpose_json"]
    assert '"chat_reply"' in row["llm_by_purpose_json"]


@pytest.mark.asyncio
async def test_proxy_stream_turn_persists_telemetry_including_the_reply_call(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers=_HEADERS,
                json={
                    "model": "atagia-memory-proxy",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Remember the proxy path."}
                    ],
                },
            ) as response:
                assert response.status_code == 200
                body = "".join([chunk async for chunk in response.aiter_text()])

    assert "data: [DONE]" in body
    rows = await _turn_rows(Path(settings.sqlite_path))
    assert len(rows) == 1
    row = rows[0]
    _assert_completed_turn(row, surface=TurnSurface.PROXY_STREAM)
    # The streamed reply is recorded by the client only when the provider
    # iterator is exhausted, which happens inside the response generator, in a
    # different async scope from the one that started the meter. Its presence
    # here is the whole point of re-binding the same meter around the stream.
    assert '"chat_reply"' in row["llm_by_purpose_json"]


def _sse_error_payload(body: str) -> dict[str, Any]:
    """Return the single SSE error object emitted in ``body``.

    Asserting on the parsed payload rather than on a substring is what makes the
    cause a contract: a substring match would still pass if the code and the
    type were dropped from the object.
    """
    errors = [
        json.loads(line.removeprefix("data: "))["error"]
        for line in body.splitlines()
        if line.startswith("data: ")
        and line != "data: [DONE]"
        and "error" in json.loads(line.removeprefix("data: "))
    ]
    assert len(errors) == 1, f"expected exactly one SSE error, got {errors}"
    return errors[0]


async def _exposed_stream_failure_state(app: Any) -> tuple[list[str], int]:
    connection = await app.state.runtime.open_connection()
    connection.row_factory = aiosqlite.Row
    try:
        cursor = await connection.execute(
            "SELECT state FROM proxy_turn_runs WHERE user_id = ?",
            ("usr_proxy_telemetry",),
        )
        run_states = [row["state"] for row in await cursor.fetchall()]
        cursor = await connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM messages
            WHERE conversation_id = ?
              AND role = 'assistant'
            """,
            ("cnv_proxy_telemetry",),
        )
        assistant_messages = (await cursor.fetchone())["total"]
    finally:
        await connection.close()
    return run_states, assistant_messages


async def _stream_proxy_turn(app: Any) -> tuple[int, str]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=_HEADERS,
            json={
                "model": "atagia-memory-proxy",
                "stream": True,
                "messages": [{"role": "user", "content": "Remember the proxy path."}],
            },
        ) as response:
            return response.status_code, "".join(
                [chunk async for chunk in response.aiter_text()]
            )


@pytest.mark.asyncio
async def test_stream_telemetry_failure_leaves_no_completed_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming surface is the one documented exception to fail-fast.

    Emission has already begun by the time the terminal commit runs, so the 200
    and the response headers are gone and the failure can only be reported in
    band, as an SSE error. What must still hold is the fail-fast OUTCOME: the
    telemetry write lives inside the terminal transaction, so a failure there
    rolls the whole turn back -- no response row, no completed run, and the run
    is left in the state that says the client saw bytes Atagia never committed.
    """
    settings = _settings(tmp_path)

    async def exploding_complete_turn_telemetry(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("telemetry write failed")

    monkeypatch.setattr(
        RetrievalEventRepository,
        "complete_turn_telemetry",
        exploding_complete_turn_telemetry,
    )
    app = create_app(settings)
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        status_code, body = await _stream_proxy_turn(app)
        run_states, assistant_messages = await _exposed_stream_failure_state(app)

    # Headers were already sent, so the failure can only be reported in band.
    assert status_code == 200
    assert "data: [DONE]" not in body
    error = _sse_error_payload(body)
    assert error["message"] == "Stream completion could not be committed"
    assert error["type"] == "atagia_stream_commit_error"
    assert error["code"] == ProxyStreamErrorCode.STREAM_COMMIT_FAILED.value
    # ...but the turn itself is not half-written: it rolled back with the fence,
    # and the run records that the client was exposed to an uncommitted answer.
    assert run_states == ["ambiguous_exposed"]
    assert assistant_messages == 0


@pytest.mark.asyncio
async def test_upstream_stream_failure_is_distinguishable_from_a_commit_failure(
    tmp_path: Path,
) -> None:
    """Two exposed failures, one run state, two causes.

    A provider that dies mid-answer and a terminal commit that fails both leave
    the run 'ambiguous_exposed' with nothing committed, so the database cannot
    tell a host which happened. The SSE payload has to, because the right
    response differs: the first never produced an answer, while the second
    produced one the user has already read.
    """
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    provider.raise_after_first_stream_event = True
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        status_code, body = await _stream_proxy_turn(app)
        run_states, assistant_messages = await _exposed_stream_failure_state(app)

    assert status_code == 200
    assert "data: [DONE]" not in body
    error = _sse_error_payload(body)
    assert error["message"] == "Upstream stream failed"
    assert error["type"] == "atagia_upstream_stream_error"
    assert error["code"] == ProxyStreamErrorCode.UPSTREAM_STREAM_FAILED.value
    # Same terminal state as the commit failure above: only the payload separates
    # them.
    assert run_states == ["ambiguous_exposed"]
    assert assistant_messages == 0
    assert (
        _PROXY_STREAM_ERROR_TYPES[ProxyStreamErrorCode.UPSTREAM_STREAM_FAILED]
        != _PROXY_STREAM_ERROR_TYPES[ProxyStreamErrorCode.STREAM_COMMIT_FAILED]
    )


@pytest.mark.asyncio
async def test_proxy_turn_without_memory_context_is_still_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fail-open turn ran without retrieval; it must not vanish from the ledger."""
    settings = _settings(tmp_path)

    async def no_context(
        service: OpenAIProxyService,
        identity: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> ProxyContextAttempt:
        # Mirror the fail-open path: the namespace exists, the context does not.
        # This stub short-circuits before any retrieval work, so the attempt
        # genuinely cost nothing and says so; the test below covers a fail-open
        # that DID spend time.
        await _ensure_capture_namespace(service, identity)
        return ProxyContextAttempt(context=None, elapsed_ms=0.0)

    monkeypatch.setattr(
        OpenAIProxyService, "_context_for_turn_fail_open", no_context
    )
    app = create_app(settings)
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_HEADERS,
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [
                        {"role": "user", "content": "Remember the proxy path."}
                    ],
                },
            )

    assert response.status_code == 200
    rows = await _turn_rows(Path(settings.sqlite_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["turn_surface"] == TurnSurface.PROXY_COMPLETION.value
    assert row["response_message_id"] is not None
    assert row["turn_to_event_write_wall_ms"] > 0.0
    # No retrieval happened, and the row says exactly that instead of pretending.
    assert row["retrieval_duration_ms"] == 0.0
    assert row["retrieval_plan_json"] == "{}"
    assert row["context_view_json"] == "{}"
    assert row["llm_total_calls"] == 1


@pytest.mark.asyncio
async def test_proxy_fail_open_reports_the_retrieval_time_it_actually_spent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fail-open is not a free turn, and the row must not claim it was.

    The turn meter is bound BEFORE the context fetch, so provider round-trips
    made before the failure are already charged to the turn. Reporting
    ``retrieval_duration_ms=0.0`` alongside them described a turn that spent
    real time in retrieval as having spent none, and nothing could catch it:
    ``0.0 <= turn_to_event_write_wall_ms`` satisfies every telemetry invariant.
    """
    settings = _settings(tmp_path)
    stall_seconds = 0.05

    async def failing_context(
        self: Any,
        user_id: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        await asyncio.sleep(stall_seconds)
        raise ConnectionError("memory context backend is unreachable")

    monkeypatch.setattr(SidecarService, "get_context", failing_context)
    app = create_app(settings)
    provider = ProxyProvider()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_HEADERS,
                json={
                    "model": "atagia-memory-proxy",
                    "messages": [
                        {"role": "user", "content": "Remember the proxy path."}
                    ],
                },
            )

    assert response.status_code == 200
    rows = await _turn_rows(Path(settings.sqlite_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["turn_surface"] == TurnSurface.PROXY_COMPLETION.value
    # The fail-open row still has no retrieval plan or context view...
    assert row["retrieval_plan_json"] == "{}"
    assert row["context_view_json"] == "{}"
    # ...but it reports what the failed attempt cost, not zero.
    assert row["retrieval_duration_ms"] >= stall_seconds * 1000.0
    assert row["retrieval_duration_ms"] <= row["turn_to_event_write_wall_ms"]
    # The reply still happened, so the turn is counted like any other.
    assert row["llm_total_calls"] == 1
    assert '"chat_reply"' in row["llm_by_purpose_json"]
    # The fail-open fallback row carries the call metrics in its trace too.
    outcome = json.loads(row["outcome_json"])
    assert outcome["memory_context_available"] is False
    assert outcome["llm_call_metrics"]["total_calls"] == 1
