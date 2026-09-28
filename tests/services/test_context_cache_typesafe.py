"""SQLite-backed cache decisions with the native finite-choice provider."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import httpx
import pytest

from atagia.app import initialize_runtime
from atagia.services.chat_service import ChatService
from atagia.services.context_cache_service import ContextCacheService
from atagia.services.llm_client import LLMClient, LLMRequestError, RetryPolicy
from atagia.services.providers.typesafe import TypeSafeProvider
from atagia.services.retrieval_service import RetrievalService
from tests.services.providers.test_typesafe import _answer_payload
from tests.services.test_context_cache_service import (
    ContextCacheProvider,
    _seed_conversation,
    _settings,
)
from tests.services.test_chat_service import ChatServiceProvider


async def test_native_cache_reuse_and_refresh_reaches_real_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = iter(["reuse", "refresh"])
    native_requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        native_requests.append(payload)
        criteria = payload["questions"]["context_reuse"]["criteria"]
        return httpx.Response(
            200,
            json=_answer_payload(
                {"context_reuse": next(decisions)},
                list(criteria),
            ),
        )

    generic = ContextCacheProvider()
    generic.name = "openai"
    retrieval_calls = 0
    original_retrieve = RetrievalService.retrieve_with_connection

    async def measured_retrieve(*args, **kwargs):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return await original_retrieve(*args, **kwargs)

    monkeypatch.setattr(RetrievalService, "retrieve_with_connection", measured_retrieve)
    settings = replace(
        _settings(tmp_path),
        llm_finite_decisions_enabled=True,
        llm_finite_decision_model="openai/decision-test-model",
        llm_component_models={
            "intent_classifier": "openai/classify-test-model",
            "context_staleness": "typesafe/jev-latest",
        },
        typesafe_api_key="test-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(
            providers=[generic, TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=1),
        )
        monkeypatch.setattr("atagia.app.build_llm_client", lambda _: client)
        runtime = await initialize_runtime(settings)
        try:
            await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
            service = ContextCacheService(runtime)

            async def resolve(message: str):
                connection = await runtime.open_connection()
                try:
                    return await service.resolve_with_connection(
                        connection,
                        user_id="usr_1",
                        conversation_id="cnv_1",
                        message_text=message,
                    )
                finally:
                    await connection.close()

            first = await resolve("Please help me debug this retry loop.")
            assert first.from_cache is False
            assert await service.publish_pending_cache_entry(
                first, last_retrieval_message_seq=1,
            )
            assert retrieval_calls == 1
            generic_before = len(generic.requests)

            followup = await resolve("Continue.")
            assert followup.from_cache is True
            assert followup.composed_context == first.composed_context
            assert len(generic.requests) == generic_before
            assert retrieval_calls == 1

            correction = await resolve("No, use the other account.")
            assert correction.from_cache is False
            assert correction.pending_cache_entry is not None
            assert retrieval_calls == 2
            assert await service.publish_pending_cache_entry(
                correction, last_retrieval_message_seq=2,
            )
            replacement = await runtime.storage_backend.get_context_view(
                str(correction.cache_key),
            )
            assert replacement is not None
            assert replacement["last_user_message_text"] == "No, use the other account."
            assert replacement["last_retrieval_message_seq"] == 2
            assert len(native_requests) == 2
            assert all(set(payload["questions"]) == {"context_reuse"} for payload in native_requests)
        finally:
            await runtime.close()


@pytest.mark.parametrize(
    "invalid_field",
    [
        "user_id",
        "effective_policy_hash",
        "cached_at",
        "version",
        "cache_revision",
    ],
)
async def test_native_service_invalid_cache_skips_http_and_runs_sqlite_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_field: str,
) -> None:
    native_calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal native_calls
        native_calls += 1
        raise AssertionError("Invalid cache must not call the native provider")

    generic = ContextCacheProvider()
    generic.name = "openai"
    settings = replace(
        _settings(tmp_path),
        llm_finite_decisions_enabled=True,
        llm_finite_decision_model="openai/decision-test-model",
        llm_component_models={
            "intent_classifier": "openai/classify-test-model",
            "context_staleness": "typesafe/jev-latest",
        },
        typesafe_api_key="test-key",
    )
    retrieval_calls = 0
    original_retrieve = RetrievalService.retrieve_with_connection

    async def measured_retrieve(*args, **kwargs):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return await original_retrieve(*args, **kwargs)

    monkeypatch.setattr(RetrievalService, "retrieve_with_connection", measured_retrieve)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(
            providers=[generic, TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=1),
        )
        monkeypatch.setattr("atagia.app.build_llm_client", lambda _: client)
        runtime = await initialize_runtime(settings)
        try:
            await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
            service = ContextCacheService(runtime)
            connection = await runtime.open_connection()
            try:
                first = await service.resolve_with_connection(
                    connection,
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    message_text="Please help me debug this retry loop.",
                )
            finally:
                await connection.close()
            assert await service.publish_pending_cache_entry(first, last_retrieval_message_seq=1)
            cache_key = str(first.cache_key)
            cached = await runtime.storage_backend.get_context_view(cache_key)
            assert cached is not None
            if invalid_field == "user_id":
                cached[invalid_field] = "another_user"
            elif invalid_field == "effective_policy_hash":
                cached[invalid_field] = "obsolete"
            elif invalid_field == "cached_at":
                cached[invalid_field] = (
                    runtime.clock.now() - timedelta(hours=1)
                ).isoformat()
            elif invalid_field == "version":
                cached[invalid_field] = 3
            else:
                cached[invalid_field] += 1
            await runtime.storage_backend.set_context_view(
                cache_key, cached, ttl_seconds=30,
            )

            connection = await runtime.open_connection()
            try:
                resolution = await service.resolve_with_connection(
                    connection,
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    message_text="Continue.",
                )
            finally:
                await connection.close()

            assert native_calls == 0
            assert retrieval_calls == 2
            assert resolution.from_cache is False
        finally:
            await runtime.close()


async def test_native_decision_keeps_final_chat_answers_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal native_calls
        native_calls += 1
        payload = json.loads(request.content)
        choice = "reuse" if native_calls == 1 else "refresh"
        criteria = payload["questions"]["context_reuse"]["criteria"]
        return httpx.Response(
            200,
            json=_answer_payload({"context_reuse": choice}, list(criteria)),
        )

    generic = ChatServiceProvider()
    generic.name = "openai"
    settings = replace(
        _settings(tmp_path),
        llm_finite_decisions_enabled=True,
        llm_finite_decision_model="openai/decision-test-model",
        llm_component_models={
            "intent_classifier": "openai/classify-test-model",
            "context_staleness": "typesafe/jev-latest",
        },
        typesafe_api_key="test-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(
            providers=[generic, TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=1),
        )
        monkeypatch.setattr("atagia.app.build_llm_client", lambda _: client)
        runtime = await initialize_runtime(settings)
        try:
            await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
            chat = ChatService(runtime)
            first = await chat.chat_reply(
                user_id="usr_1", conversation_id="cnv_1",
                message_text="Please help me debug this retry loop.",
                assistant_mode_id="coding_debug",
            )
            followup = await chat.chat_reply(
                user_id="usr_1", conversation_id="cnv_1",
                message_text="Continue.", assistant_mode_id="coding_debug",
            )
            correction = await chat.chat_reply(
                user_id="usr_1", conversation_id="cnv_1",
                message_text="No, use the other account.",
                assistant_mode_id="coding_debug",
            )
            assert [result.response_text for result in (first, followup, correction)] == [
                "Check the retry guard first.",
            ] * 3
            assert native_calls == 2
            assert first.retrieval_event_id != followup.retrieval_event_id
            assert correction.retrieval_event_id != followup.retrieval_event_id
        finally:
            await runtime.close()


@pytest.mark.parametrize("cancelled", [False, True])
async def test_native_service_failure_or_cancellation_does_not_retrieve_or_replace_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancelled: bool,
) -> None:
    native_calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal native_calls
        native_calls += 1
        if cancelled:
            raise asyncio.CancelledError
        return httpx.Response(401)

    generic = ContextCacheProvider()
    generic.name = "openai"
    settings = replace(
        _settings(tmp_path),
        llm_finite_decisions_enabled=True,
        llm_component_models={
            "intent_classifier": "openai/classify-test-model",
            "context_staleness": "typesafe/jev-latest",
        },
        typesafe_api_key="test-key",
    )
    retrieval_calls = 0
    original_retrieve = RetrievalService.retrieve_with_connection

    async def measured_retrieve(*args, **kwargs):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return await original_retrieve(*args, **kwargs)

    monkeypatch.setattr(RetrievalService, "retrieve_with_connection", measured_retrieve)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(
            providers=[generic, TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=1),
        )
        monkeypatch.setattr("atagia.app.build_llm_client", lambda _: client)
        runtime = await initialize_runtime(settings)
        try:
            await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
            service = ContextCacheService(runtime)
            connection = await runtime.open_connection()
            try:
                first = await service.resolve_with_connection(
                    connection,
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    message_text="Please help me debug this retry loop.",
                )
            finally:
                await connection.close()
            assert await service.publish_pending_cache_entry(first, last_retrieval_message_seq=1)
            assert retrieval_calls == 1
            cache_key = str(first.cache_key)
            original_entry = await runtime.storage_backend.get_context_view(cache_key)

            connection = await runtime.open_connection()
            try:
                with pytest.raises(asyncio.CancelledError if cancelled else LLMRequestError):
                    await service.resolve_with_connection(
                        connection,
                        user_id="usr_1",
                        conversation_id="cnv_1",
                        message_text="No, use the other account.",
                    )
            finally:
                await connection.close()
            assert native_calls == 1
            assert retrieval_calls == 1
            assert await runtime.storage_backend.get_context_view(cache_key) == original_entry
        finally:
            await runtime.close()
