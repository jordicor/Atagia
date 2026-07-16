"""End-to-end durable transcript and replay coverage for the OpenAI proxy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from atagia.app import create_app
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
)
from tests.api.test_openai_proxy import ProxyProvider, _settings


def _headers(
    *,
    conversation_id: str,
    request_id: str | None = None,
    response_id: str | None = None,
    request_seq: int | None = None,
    response_seq: int | None = None,
) -> dict[str, str]:
    headers = {
        "Authorization": "Bearer service-key",
        "X-Atagia-User-Id": "usr_proxy_transcript",
        "X-Atagia-Platform-Id": "proxy_tests",
        "X-Atagia-Conversation-Id": conversation_id,
    }
    if request_id is not None:
        headers["X-Atagia-Message-Id"] = request_id
    if response_id is not None:
        headers["X-Atagia-Response-Message-Id"] = response_id
    if request_seq is not None:
        headers["X-Atagia-Source-Seq"] = str(request_seq)
    if response_seq is not None:
        headers["X-Atagia-Response-Source-Seq"] = str(response_seq)
    return headers


def _body(content: str = "Remember the durable proxy turn.") -> dict[str, Any]:
    return {
        "model": "atagia-memory-proxy",
        "messages": [{"role": "user", "content": content}],
    }


def _chat_requests(provider: ProxyProvider) -> list[LLMCompletionRequest]:
    return [
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "chat_reply"
    ]


@pytest.mark.asyncio
async def test_completed_pair_replays_across_stream_framing_without_context_or_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_calls = 0

    async def context_without_retrieval(self: object, **kwargs: Any) -> None:
        nonlocal context_calls
        context_calls += 1
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        context_without_retrieval,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    headers = _headers(
        conversation_id="cnv_replay",
        request_id="msg_replay_request",
        response_id="msg_replay_response",
    )
    body = _body()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(
            provider_name=provider.name,
            providers=[provider],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/v1/chat/completions", headers=headers, json=body
            )
            connection = await app.state.runtime.open_connection()
            try:
                await connection.execute(
                    "UPDATE conversations SET status = 'closed' WHERE id = ?",
                    ("cnv_replay",),
                )
                await connection.commit()
            finally:
                await connection.close()
            replay_body = {
                **body,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            replay = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json=replay_body,
            )

        connection = await app.state.runtime.open_connection()
        try:
            messages = await (
                await connection.execute(
                    "SELECT id, role FROM messages ORDER BY seq ASC"
                )
            ).fetchall()
            run = await (
                await connection.execute(
                    "SELECT * FROM proxy_turn_runs WHERE request_message_id = ?",
                    ("msg_replay_request",),
                )
            ).fetchone()
            assert run is not None
            job_ids = json.loads(run["durable_job_ids_json"])
            stored_jobs = await (
                await connection.execute(
                    "SELECT job_id FROM worker_job_runs WHERE job_id IN ({})".format(
                        ",".join("?" for _ in job_ids)
                    ),
                    tuple(job_ids),
                )
            ).fetchall()
        finally:
            await connection.close()

    assert first.status_code == 200
    assert replay.status_code == 200
    assert "Proxy reply." in replay.text
    assert "data: [DONE]" in replay.text
    assert context_calls == 1
    assert len(_chat_requests(provider)) == 1
    assert [(row["id"], row["role"]) for row in messages] == [
        ("msg_replay_request", "user"),
        ("msg_replay_response", "assistant"),
    ]
    assert run["state"] == "completed"
    assert job_ids
    assert {row["job_id"] for row in stored_jobs} == set(job_ids)


@pytest.mark.asyncio
async def test_id_pair_matrix_and_no_id_requests_have_new_turn_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_context(self: object, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        no_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    body = _body()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            request_only = await client.post(
                "/v1/chat/completions",
                headers=_headers(
                    conversation_id="cnv_pair_matrix",
                    request_id="msg_request_only",
                ),
                json=body,
            )
            response_only = await client.post(
                "/v1/chat/completions",
                headers=_headers(
                    conversation_id="cnv_pair_matrix",
                    response_id="msg_response_only",
                ),
                json=body,
            )
            no_ids_first = await client.post(
                "/v1/chat/completions",
                headers=_headers(conversation_id="cnv_pair_matrix"),
                json=body,
            )
            no_ids_second = await client.post(
                "/v1/chat/completions",
                headers=_headers(conversation_id="cnv_pair_matrix"),
                json=body,
            )
        connection = await app.state.runtime.open_connection()
        try:
            message_count = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) AS count FROM messages"
                        )
                    ).fetchone()
                )["count"]
            )
            run_count = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) AS count FROM proxy_turn_runs"
                        )
                    ).fetchone()
                )["count"]
            )
        finally:
            await connection.close()

    assert request_only.status_code == 400
    assert response_only.status_code == 400
    assert request_only.json()["error"]["code"] == "incomplete_message_id_pair"
    assert response_only.json()["error"]["code"] == "incomplete_message_id_pair"
    assert no_ids_first.status_code == no_ids_second.status_code == 200
    assert len(_chat_requests(provider)) == 2
    assert message_count == 4
    assert run_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    (
        {"temperature": 0.7},
        {"messages": [{"role": "user", "content": "Changed text"}]},
        {"tool_choice": "none"},
        {"seed": 42},
    ),
)
async def test_response_determining_client_changes_conflict_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any],
) -> None:
    async def no_context(self: object, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        no_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    headers = _headers(
        conversation_id="cnv_fingerprint",
        request_id="msg_fingerprint_request",
        response_id="msg_fingerprint_response",
    )
    original = _body()
    changed = {**original, **mutation}
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/v1/chat/completions", headers=headers, json=original
            )
            conflict = await client.post(
                "/v1/chat/completions", headers=headers, json=changed
            )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "proxy_request_fingerprint_conflict"
    assert len(_chat_requests(provider)) == 1


class MultiToolProvider(ProxyProvider):
    def __init__(self) -> None:
        super().__init__()
        self.chat_call_count = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if request.metadata.get("purpose") != "chat_reply":
            return await super().complete(request)
        self.requests.append(request)
        self.chat_call_count += 1
        if self.chat_call_count == 1:
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="",
                tool_calls=[
                    {
                        "id": "call_alpha",
                        "type": "function",
                        "name": "lookup",
                        "arguments": {"query": "alpha"},
                    },
                    {
                        "id": "call_beta",
                        "type": "function",
                        "name": "lookup",
                        "arguments": '{"query":"beta"}',
                    },
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text="Both results were processed.",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_multi_tool_continuation_persists_one_lossless_causal_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_context(self: object, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        no_context,
    )
    app = create_app(_settings(tmp_path))
    provider = MultiToolProvider()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        }
    ]
    first_body = {
        **_body("Call both tools."),
        "tools": tools,
        "tool_choice": "auto",
    }
    first_headers = _headers(
        conversation_id="cnv_multi_tool",
        request_id="msg_tool_request_1",
        response_id="msg_tool_response_1",
    )
    second_headers = _headers(
        conversation_id="cnv_multi_tool",
        request_id="msg_tool_request_2",
        response_id="msg_tool_response_2",
    )
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/v1/chat/completions", headers=first_headers, json=first_body
            )
            assert first.status_code == 200, first.text
            calls = first.json()["choices"][0]["message"]["tool_calls"]
            second_body = {
                "model": "atagia-memory-proxy",
                "messages": [
                    {"role": "user", "content": "Call both tools."},
                    {
                        "role": "assistant",
                        "content": None,
                        "message_id": "msg_tool_response_1",
                        "tool_calls": calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_beta",
                        "name": "lookup",
                        "content": "",
                        "status": "error",
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_alpha",
                        "name": "lookup",
                        "content": {"items": [1, 2], "ok": True},
                    },
                ],
                "tools": tools,
            }
            second = await client.post(
                "/v1/chat/completions", headers=second_headers, json=second_body
            )
            replay = await client.post(
                "/v1/chat/completions", headers=second_headers, json=second_body
            )
        connection = await app.state.runtime.open_connection()
        try:
            rows = await (
                await connection.execute(
                    """
                    SELECT id, role, text, metadata_json,
                           active_presence_id, source_presence_id
                    FROM messages
                    ORDER BY seq ASC
                    """
                )
            ).fetchall()
            job_rows = await (
                await connection.execute(
                    """
                    SELECT recovery_envelope_json
                    FROM worker_job_runs
                    WHERE json_extract(
                        recovery_envelope_json,
                        '$.payload.message_id'
                    ) IN (
                        'msg_tool_request_1',
                        'msg_tool_response_1',
                        'msg_tool_request_2',
                        'msg_tool_response_2'
                    )
                    """
                )
            ).fetchall()
        finally:
            await connection.close()

    assert first.status_code == second.status_code == replay.status_code == 200, (
        second.text,
        replay.text,
    )
    assert provider.chat_call_count == 2
    assert [row["id"] for row in rows] == [
        "msg_tool_request_1",
        "msg_tool_response_1",
        "msg_tool_request_2",
        "msg_tool_response_2",
    ]
    assert rows[1]["role"] == "assistant"
    assert "call_alpha" in rows[1]["text"] and "call_beta" in rows[1]["text"]
    assistant_transcript = json.loads(rows[1]["metadata_json"])[
        "atagia_proxy_transcript"
    ]
    stored_calls = assistant_transcript["tool_projection"]["calls"]
    assert stored_calls[0]["arguments"] == {"query": "alpha"}
    assert stored_calls[1]["arguments"] == '{"query":"beta"}'
    assert assistant_transcript["replay"]["tool_calls"] == calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "alpha"}
    assert rows[2]["role"] == "tool"
    assert [row["active_presence_id"] for row in rows] == [
        "default_assistant",
        "default_assistant",
        "default_assistant",
        "default_assistant",
    ]
    assert [row["source_presence_id"] for row in rows] == [
        "human_owner",
        "default_assistant",
        "default_assistant",
        "default_assistant",
    ]
    job_source_presence: dict[str, set[str]] = {}
    for job_row in job_rows:
        envelope = json.loads(job_row["recovery_envelope_json"])
        payload = envelope["payload"]
        job_source_presence.setdefault(payload["message_id"], set()).add(
            payload["source_presence_id"]
        )
    assert job_source_presence == {
        "msg_tool_request_1": {"human_owner"},
        "msg_tool_response_1": {"default_assistant"},
        "msg_tool_request_2": {"default_assistant"},
        "msg_tool_response_2": {"default_assistant"},
    }
    transcript = json.loads(rows[2]["metadata_json"])["atagia_proxy_transcript"]
    assert transcript["tool_projection"]["parent_response_message_id"] == (
        "msg_tool_response_1"
    )
    results = transcript["tool_projection"]["results"]
    assert [result["tool_call_id"] for result in results] == [
        "call_beta",
        "call_alpha",
    ]
    assert results[0]["content"] == ""
    assert results[0]["extra"] == {"status": "error"}
    assert results[1]["content"] == {"items": [1, 2], "ok": True}


@pytest.mark.asyncio
async def test_exposed_stream_is_permanently_ambiguous_and_never_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_context(self: object, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        no_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    provider.raise_after_first_stream_event = True
    headers = _headers(
        conversation_id="cnv_ambiguous_stream",
        request_id="msg_ambiguous_request",
        response_id="msg_ambiguous_response",
    )
    body = {**_body(), "stream": True}
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/v1/chat/completions", headers=headers, json=body
            )
            retry = await client.post(
                "/v1/chat/completions", headers=headers, json=body
            )
        connection = await app.state.runtime.open_connection()
        try:
            run = await (
                await connection.execute(
                    "SELECT state FROM proxy_turn_runs WHERE request_message_id = ?",
                    ("msg_ambiguous_request",),
                )
            ).fetchone()
            response_row = await (
                await connection.execute(
                    "SELECT id FROM messages WHERE id = ?",
                    ("msg_ambiguous_response",),
                )
            ).fetchone()
        finally:
            await connection.close()

    assert first.status_code == 200
    assert "atagia_upstream_stream_error" in first.text
    assert "data: [DONE]" not in first.text
    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "stream_retry_requires_new_ids"
    assert len(_chat_requests(provider)) == 1
    assert run["state"] == "ambiguous_exposed"
    assert response_row is None


@pytest.mark.asyncio
async def test_stream_terminal_commit_failure_emits_no_terminal_or_done_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_context(self: object, **kwargs: Any) -> None:
        return None

    async def fail_terminal_commit(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected terminal commit failure")

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        no_context,
    )
    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.finalize_proxy_turn",
        fail_terminal_commit,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    headers = _headers(
        conversation_id="cnv_terminal_failure",
        request_id="msg_terminal_failure_request",
        response_id="msg_terminal_failure_response",
    )
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={**_body(), "stream": True},
            )
        connection = await app.state.runtime.open_connection()
        try:
            run = await (
                await connection.execute(
                    "SELECT state FROM proxy_turn_runs WHERE request_message_id = ?",
                    ("msg_terminal_failure_request",),
                )
            ).fetchone()
            response_row = await (
                await connection.execute(
                    "SELECT id FROM messages WHERE id = ?",
                    ("msg_terminal_failure_response",),
                )
            ).fetchone()
            job_count = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) AS count FROM worker_job_runs"
                        )
                    ).fetchone()
                )["count"]
            )
        finally:
            await connection.close()

    assert response.status_code == 200
    assert "Stream completion could not be committed" in response.text
    assert '"finish_reason":"stop"' not in response.text
    assert "data: [DONE]" not in response.text
    assert run["state"] == "ambiguous_exposed"
    assert response_row is None
    assert job_count == 0


@pytest.mark.asyncio
async def test_terminal_rollback_includes_icp_generation_and_all_durable_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atagia.services.proxy_turn_service import (
        finalize_proxy_turn as real_finalize_proxy_turn,
    )

    async def no_context(self: object, **kwargs: Any) -> None:
        return None

    async def fail_after_response(*args: Any, **kwargs: Any):
        def failpoint(name: str) -> None:
            if name == "response_inserted":
                raise RuntimeError("injected after response insert")

        return await real_finalize_proxy_turn(
            *args,
            **kwargs,
            failpoint=failpoint,
        )

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        no_context,
    )
    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.finalize_proxy_turn",
        fail_after_response,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    headers = _headers(
        conversation_id="cnv_atomic_generation_rollback",
        request_id="msg_atomic_generation_request",
        response_id="msg_atomic_generation_response",
    )
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json=_body(),
            )
        connection = await app.state.runtime.open_connection()
        try:
            run = await (
                await connection.execute(
                    "SELECT state FROM proxy_turn_runs WHERE request_message_id = ?",
                    ("msg_atomic_generation_request",),
                )
            ).fetchone()
            response_row = await (
                await connection.execute(
                    "SELECT id FROM messages WHERE id = ?",
                    ("msg_atomic_generation_response",),
                )
            ).fetchone()
            job_count = int(
                (
                    await (
                        await connection.execute(
                            "SELECT COUNT(*) AS count FROM worker_job_runs"
                        )
                    ).fetchone()
                )["count"]
            )
            lifecycle = await (
                await connection.execute(
                    "SELECT icp_refresh_generation FROM user_lifecycles "
                    "WHERE user_id = ?",
                    ("usr_proxy_transcript",),
                )
            ).fetchone()
        finally:
            await connection.close()

    assert response.status_code == 500
    assert run["state"] == "generating"
    assert response_row is None
    assert job_count == 0
    assert lifecycle["icp_refresh_generation"] == 0


@pytest.mark.asyncio
async def test_completed_turn_survives_dispatch_crash_and_restart_without_duplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_context(self: object, **kwargs: Any) -> None:
        return None

    dispatch_calls = 0

    async def crash_before_dispatch(*args: Any, **kwargs: Any) -> None:
        nonlocal dispatch_calls
        dispatch_calls += 1
        raise RuntimeError("simulated crash before transient dispatch")

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        no_context,
    )
    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.dispatch_proxy_terminal_jobs",
        crash_before_dispatch,
    )
    settings = _settings(tmp_path)
    headers = _headers(
        conversation_id="cnv_dispatch_restart",
        request_id="msg_dispatch_restart_request",
        response_id="msg_dispatch_restart_response",
    )
    body = _body()

    first_app = create_app(settings)
    first_provider = ProxyProvider()
    async with first_app.router.lifespan_context(first_app):
        first_app.state.runtime.llm_client = LLMClient(
            first_provider.name,
            [first_provider],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=first_app),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/v1/chat/completions", headers=headers, json=body
            )
        connection = await first_app.state.runtime.open_connection()
        try:
            first_job_ids = [
                row["job_id"]
                for row in await (
                    await connection.execute(
                        "SELECT job_id FROM worker_job_runs ORDER BY job_id"
                    )
                ).fetchall()
            ]
            run = await (
                await connection.execute(
                    "SELECT state, durable_job_ids_json FROM proxy_turn_runs "
                    "WHERE request_message_id = ?",
                    ("msg_dispatch_restart_request",),
                )
            ).fetchone()
        finally:
            await connection.close()

    second_app = create_app(settings)
    second_provider = ProxyProvider()
    async with second_app.router.lifespan_context(second_app):
        second_app.state.runtime.llm_client = LLMClient(
            second_provider.name,
            [second_provider],
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=second_app),
            base_url="http://testserver",
        ) as client:
            replay = await client.post(
                "/v1/chat/completions", headers=headers, json=body
            )
        connection = await second_app.state.runtime.open_connection()
        try:
            replay_job_ids = [
                row["job_id"]
                for row in await (
                    await connection.execute(
                        "SELECT job_id FROM worker_job_runs ORDER BY job_id"
                    )
                ).fetchall()
            ]
        finally:
            await connection.close()

    assert first.status_code == replay.status_code == 200
    assert run["state"] == "completed"
    assert first_job_ids
    assert set(json.loads(run["durable_job_ids_json"])) == set(first_job_ids)
    assert replay_job_ids == first_job_ids
    assert len(_chat_requests(first_provider)) == 1
    assert _chat_requests(second_provider) == []
    assert dispatch_calls == 1


class BlockingProxyProvider(ProxyProvider):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if request.metadata.get("purpose") != "chat_reply":
            return await super().complete(request)
        self.requests.append(request)
        self.entered.set()
        await self.release.wait()
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text="Owned completion.",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_concurrent_compatible_retry_gets_retryable_409_and_one_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_context(self: object, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        no_context,
    )
    app = create_app(_settings(tmp_path))
    provider = BlockingProxyProvider()
    headers = _headers(
        conversation_id="cnv_concurrent_pair",
        request_id="msg_concurrent_request",
        response_id="msg_concurrent_response",
    )
    body = _body()
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            owner_task = asyncio.create_task(
                client.post("/v1/chat/completions", headers=headers, json=body)
            )
            await asyncio.wait_for(provider.entered.wait(), timeout=5)
            competitor = await client.post(
                "/v1/chat/completions", headers=headers, json=body
            )
            provider.release.set()
            owner = await asyncio.wait_for(owner_task, timeout=10)

    assert owner.status_code == 200
    assert competitor.status_code == 409
    assert competitor.json()["error"]["code"] == "request_in_progress"
    assert competitor.headers["Retry-After"] == "1"
    assert len(_chat_requests(provider)) == 1


@pytest.mark.asyncio
async def test_sequence_claims_and_json_looking_strings_remain_literal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_context: list[str] = []

    async def capture_context(self: object, **kwargs: Any) -> None:
        captured_context.append(kwargs["message"])
        return None

    monkeypatch.setattr(
        "atagia.services.openai_proxy_service.SidecarService.get_context",
        capture_context,
    )
    app = create_app(_settings(tmp_path))
    provider = ProxyProvider()
    literal = ' {"looks":"json", "spaces":  [1,  2], "escaped":"\\n"} '
    headers = _headers(
        conversation_id="cnv_sequences",
        request_id="msg_sequence_request",
        response_id="msg_sequence_response",
        request_seq=7,
        response_seq=8,
    )
    body = _body(literal)
    async with app.router.lifespan_context(app):
        app.state.runtime.llm_client = LLMClient(provider.name, [provider])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/v1/chat/completions", headers=headers, json=body
            )
            replay = await client.post(
                "/v1/chat/completions",
                headers=_headers(
                    conversation_id="cnv_sequences",
                    request_id="msg_sequence_request",
                    response_id="msg_sequence_response",
                ),
                json=body,
            )
            bad_sequence = await client.post(
                "/v1/chat/completions",
                headers=_headers(
                    conversation_id="cnv_sequences",
                    request_id="msg_sequence_request",
                    response_id="msg_sequence_response",
                    response_seq=9,
                ),
                json=body,
            )

    assert first.status_code == replay.status_code == 200
    assert bad_sequence.status_code == 409
    assert bad_sequence.json()["error"]["code"] == "source_sequence_conflict"
    assert captured_context == [literal]
    assert _chat_requests(provider)[0].messages[-1].content == literal
