"""Native context reuse decisions through the production scorer and provider."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json

import httpx
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.memory.context_staleness import ContextStalenessScorer
from atagia.services.llm_client import LLMClient, LLMError, LLMRequestError, RetryPolicy
from atagia.services.providers.typesafe import TypeSafeProvider
from tests.memory.test_context_staleness import _entry_payload, _request_payload, _resolved_policy
from tests.services.providers.test_typesafe import _answer_payload


def _settings() -> Settings:
    return replace(
        Settings.from_env({}),
        llm_finite_decisions_enabled=True,
    )


def _scorer(client: LLMClient[object], *, minute: int = 1) -> ContextStalenessScorer:
    return ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, minute, tzinfo=timezone.utc)),
        llm_client=client,
        settings=_settings(),
    )


@pytest.mark.parametrize(
    ("message", "choice", "expected_refresh"),
    [
        ("Continue with the login test.", "reuse", False),
        ("No, the other account.", "refresh", True),
        ("Actually, use the previous account.", "refresh", True),
        ("What about the deployment instead?", "refresh", True),
    ],
)
async def test_native_choice_controls_ambiguous_and_short_messages(
    message: str, choice: str, expected_refresh: bool,
) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        question = payload["questions"]["context_reuse"]
        return httpx.Response(
            200,
            json=_answer_payload({"context_reuse": choice}, list(question["criteria"])),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(
            providers=[TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=1),
        )
        score = await _scorer(client).score(
            _entry_payload(),
            _request_payload(message_text=message),
            _resolved_policy(),
        )

    assert len(requests) == 1
    assert set(requests[0]["questions"]) == {"context_reuse"}
    assert set(requests[0]["questions"]["context_reuse"]["criteria"]) == {"reuse", "refresh"}
    assert score.decision_band == "native_choice"
    assert score.should_refresh is expected_refresh
    assert score.matched_signals == [f"native_{choice}"]
    if choice == "reuse":
        assert score.hard_sync is False
        assert score.staleness < score.effective_sync_threshold
    else:
        assert score.hard_sync is True


@pytest.mark.parametrize(
    ("entry_update", "request_update", "expected_signal"),
    [
        ({"user_id": "another_user"}, {}, "user_id_mismatch"),
        ({"effective_policy_hash": "obsolete"}, {}, "effective_policy_hash_mismatch"),
        ({"version": 3}, {}, "cache_entry_validation_failed"),
        ({"cached_at": "2026-04-02T11:00:00+00:00"}, {}, "time_ceiling_exceeded"),
        ({}, {"cache_enabled": False}, "cache_disabled"),
    ],
)
async def test_hard_prechecks_avoid_native_http(
    entry_update: dict[str, object],
    request_update: dict[str, object],
    expected_signal: str,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("A hard precheck must not call the provider")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        score = await _scorer(client).score(
            {**_entry_payload(), **entry_update},
            {**_request_payload(message_text="No, change that."), **request_update},
            _resolved_policy(),
        )

    assert calls == 0
    assert score.should_refresh is True
    assert score.decision_band == "hard_precheck"
    assert expected_signal in score.matched_signals


async def test_decisive_age_penalty_refreshes_before_native_http() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("A decisive age penalty must not call the provider")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        score = await _scorer(client, minute=17).score(
            _entry_payload(),
            _request_payload(message_text="Continue."),
            _resolved_policy(),
        )

    assert calls == 0
    assert score.should_refresh is True
    assert score.decision_band == "hard_precheck"
    assert "base_penalty_threshold" in score.matched_signals


@pytest.mark.parametrize("response", [httpx.Response(401), httpx.Response(200, json={"answers": {}})])
async def test_native_provider_errors_fail_explicitly(response: httpx.Response) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response),
    ) as http:
        client = LLMClient(
            providers=[TypeSafeProvider("test-key", client=http)],
            retry_policy=RetryPolicy(attempts=1),
        )
        with pytest.raises((LLMRequestError, LLMError)):
            await _scorer(client).score(
                _entry_payload(),
                _request_payload(message_text="No, change that."),
                _resolved_policy(),
            )


async def test_native_provider_cancellation_propagates() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = LLMClient(providers=[TypeSafeProvider("test-key", client=http)])
        with pytest.raises(asyncio.CancelledError):
            await _scorer(client).score(
                _entry_payload(),
                _request_payload(message_text="No, change that."),
                _resolved_policy(),
            )
